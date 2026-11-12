我已经通读了这篇《VoiceBench 推理代码深入拆解》。这份文档质量很高（每个论点都带 文件:行号 锚点）。当前工作区里没有 mola_duplex_training 的源码，所以我以文档为准，帮你把推理代码的运行时执行流从头到尾串一遍——不按文档的章节顺序，而是按"程序跑起来后实际发生了什么"的时间顺序讲。

一、运行时全景：一条 wav 的生命周期
程序入口是 inference_streaming_fast.py（下称 fast）的 main()（fast:1645-2050），对每条 wav 调用一次 generate()（fast:426）。整体是这样转的：

main()
 ├─ build_model()            ← 一次性：装模型（只做一次，后面 7783 条复用）
 └─ for each wav:
     ├─ precompute_all_audio()   ← 把整条音频提前算成特征组
     ├─ generate()               ← 核心主循环（两阶段状态机）
     │    ├─ Phase 1: 有真音频 → 控制态用投机预填充，开口了切逐拍 AR
     │    └─ Phase 2: 音频耗尽 → 喂零波形"安静陪跑"，直到说完/超时
     └─ _contract_entry() + 原子写盘  ← 产出档案
下面逐段拆。

二、启动阶段：build_model 的双目录装配（只做一次）
这是最容易踩坑的一段（文档 §2，ref:523-660）。它分两个目录：

--model_path（基座 66GB）= 图纸：只提供 config.json 和 tokenizer
--ckpt（训练产物 59GB）= 发动机：只提供权重
关键时序：

先在 cuda 上"空转装配"（ref:587-598）：weights_path=None 空构建。文档特别强调这里有个实测踩出来的坑——如果走 meta-device 路径，rotary inv_freq、audio positional embedding 这些运行时计算的 buffer 会和 cuda 构建差最后一位浮点数，在 bf16 + MoE 自由解码里足以翻转个别 token。所以必须空 cuda 构建 + 事后 overlay，才能和 DCP 路径 bit 一致。
三道防伪检查（ref:608-632）：合并 ckpt 里的 cadence/layout 印记 → check_encoder_cadence 比对节拍（本 ckpt 是 240ms，传错 env 会直接拦下）→ assert_layout_compatible 校验布局 → 冻结音频塔。
权重覆盖（ref:644-648）：检测到 model.safetensors.index.json 就走 hf_direct 事后 overlay；否则走 DCP 加载。
disable_talker()（ref:602）：只要文本输出，语音合成侧关掉。
三、每条音频开始前：precompute_all_audio（fast:322）
传统做法是一边推理一边算音频特征。fast 的做法是提前把整条音频按参考脚本一模一样的 push 序列算完，收成 audio_row_groups，每项是 [3, hidden] 的张量。

为什么可以这么做还保证 bit 级一致（fast:328-331 注释）：

StreamingAudioEncoder.push 是因果的——当前行的特征不偷看未来；
参考脚本里交错执行的 LM forward 不影响音频编码器状态。
这里要分清文档 §4 的三层数字（很多人混在一起）：

层	值	角色
推波形节拍
永远 80ms
把 wav 切成 80ms 一口喂给 mel 前端
mel 行节奏
12.5 Hz
每 80ms 出一行特征
tower 分组
本 ckpt = 240ms
音频塔攒 3 行才算一次 forward（WS4 提速点）
注意：cadence=240 只改 tower 的计算批次，不改输出节奏——模型每"拍"（chunk）恒为 240ms，固定收到 3 行听觉特征。剩下的零头（不足 3 行）记日志后丢弃。

四、核心：generate() 主循环——两阶段状态机（fast:1150-1261）
这是"推理到底怎么运行"的心脏。它是一台状态机驱动的循环，每一"拍"（240ms）做一次决策。

4.1 每拍的 token 布局（先理解"表格"长什么样）
每一拍，模型看到的输入是这样一个 chunk（fast:367 + ref:907）：

<|chunk_bos|> <|audio_start|> <audio>×3 <|audio_end|> <|agent_text_start|>   ← 输入部分（partial）
                                                                    ↑
                                              模型在"这个位置的前一格"思考，
                                              预测决策格里该填什么
forward_step（ref:988）会用 audio_mask 把预计算好的 3 行音频特征精确替换到 <|audio|> 占位符的 embedding 上，数量对不上直接 raise（ref:1020-1023）。

补全（completion）时（v2 模式，ref:968）：

[STATE_token][text_token?] <|agent_text_end|> <|action_start|> <|action_end|> <|chunk_eos|>
也就是说：每拍模型先从 {听、开口、被打断} 里选一个"行为动词"，选"开口"才接着吐字。（文档 §3 的表格：<|listen|> id=151695 是主状态，<|interrupt|> id=151701 是打断事件）

4.2 Phase 1：源音频没放完（fast:1162）
while 音频未耗尽:
    if 当前处于控制态(listen/idle/None):
        → prefill_control_run()   # 投机：一口气草拟 K 拍，批量验收（下面详述）
    if 出现转折(要说话/被打断):
        → ar_one_chunk()          # 逐拍自回归，一字一字来
        直到说完回到控制态
    # text_break 在 Phase 1 不停——继续吃音频（参考行为）
铁律：每拍无论状态都吃一组音频（听的时候吃，说的时候也吃，fast:607-610）。这是 barge-in（随时插话）能力的关键——顺序永远是"consume audio → construct chunk → predict"，耳朵全程开着。

4.3 Phase 2：音频放完了（fast:1183）
音频耗尽后不是直接停，而是用数字零波形继续过同一前端产生 pad 组，每拍照常"吃安静"。

为什么（ref:142-145）：训练是 always-on 的（每拍都带音频行），推理时突然断音频 = 喂给模型一个训练从未见过的分布（OOD）。所以用零波形模拟"对方说完后的安静"。

停止条件：至少安静 PAD_MIN_CHUNKS=21 拍（5 秒，ceil(5.0/0.24)）且不在回答中途；或出现 turn-end 信号。上限 max_chunks=200（只封 Phase 2，≈48s，Phase 1 听段不受限）。

4.4 电话类比
文档 §6 的大白话很传神：Phase 1 = 你说的时候对方边听边可能插话；Phase 2 = 你说完了，对方不立刻挂断，在安静里等 5 秒看你是否还有下文；回答没说完就一直陪着。

五、提速的灵魂：投机预填充 prefill_control_run（fast:812-1027）
这是 fast 相对参考实现快 3~8 倍的原因（benchmark/result.md:20），也是这份文档最精彩的一节。

5.1 旧方法有多浪费
参考脚本 use_cache=False（inference_streaming.py:603、ref:1049）：每来一拍，把全部历史前缀从头重算一遍。听 23 拍 = 23 次 forward，第 23 次的前缀已涨到 ~4000 token。

5.2 投机五步
核心洞察："听"的时候十有八九下一拍还是"听"。

草拟（fast:804）：控制态下草拟 K 个未来 chunk，决策格全填"假设继续听"。
K 的三重上限（fast:827-840）：min(64, 剩余音频组数, 16384 token 预算内塞得下的数量)；Phase 2 再加 pad/max_chunks 余量限制。
一次大 forward（fast:887-899）：[已提交前缀 + K 个草拟 chunk] 一次算完，拿到全序列 logits。
从左到右逐个验收（fast:911-940）：
位置 i 的 top1 和草拟一致 → 原样提交，白赚；
第一个不一致处 = turn point：位置 i 的预测依然有效（它只依赖确认过的前 i-1 拍 + 本拍自己的输入，"决策格是被预测的对象，不是它自己预测的输入"），用真实路由结果重建第 i 拍提交，i+1 之后全部丢弃；
v2 下若 turn point 是"开口"，还要执行"二次 forward dance"取首词（下面 §5.4）。
丢弃零浪费（fast:862-865）：草拟用的音频/pad 行是缓存且确定性的，重复草拟直接复用。
全 K 验收通过 → 返回 None，外层继续草拟下一轮。

5.3 为什么这么做是对的（不是近似）
thinker 是 is_causal=True 的因果注意力 + 音频前端也完全因果
⇒ "大 forward 读位置 i" 与 "参考逐拍跑位置 i" 数学上等价
残差只有 GPU 归约形状不同带来的 ~1e-6 浮点噪声，几乎不可能翻转 argmax
且有 --verify_ar 实测背书（§六）
效果对比（文档 §8.2 例子）：一条 5.65s 音频 23 拍听段，旧法 23 次 forward，新法 1 次大 forward + 转折提交；全量 80s/条 → 10s/条。

5.4 AR 路径与"一拍两用"（fast:1028-1148）
进入说话状态就没捷径了——文本 token 互相依赖，只能逐拍自回归：

partial 进去，forward 一次，在 <|agent_text_start|> 位置读 top1 → 路由；
若路由结果是 speak：再 forward 一次（这次输入里带上 state_token），在下一个位置读 top5 → pick_text_state_then_text（duplex_routing.py:345）选出第一个文本词；
提交：[speak][首词] + 收尾四件套——这一拍落账为 "text" 拍但携带首词（这就是 FAQ 里"speak 去哪了"的答案：v2 里 speak 拍落账成 text，token_stream 里保留 "speak" 痕迹）。
六、路由状态机：route_position_state_then_text（duplex_routing.py:367）
每拍先问"我是谁"再定内容。返回值 new_state ∈ {LISTEN, IDLE, SPEAK, INTERRUPT}——永不返回 TEXT（text 拍的图状态就是 SPEAK，contract 层再把 SPEAK+text 映射为 "text"）。

关键规则：

控制态下：top1 是 interrupt 且转移合法 → I；是 speak → SPEAK；是 idle → IDLE；
说话中：SPEAK→SPEAK 继续；→ LISTEN/IDLE 表示说完；acceptance 模式下 top1=interrupt → SPEAK→I（被打断截断）；
噪声容错：说话中决策位 top1 若是普通文本 token（模型"口误"），路由器不崩溃，按"继续说"处理并重选文本——"never a state-machine stall"；
转移表由 DuplexTransitionGraph（fast:489-537）声明式定义，每次落账前 _assert_routing_invariants（fast:157）做硬断言，违章直接 raise。
还有一个 accepted_user 闩锁（fast:673-691）：收到 <|interrupt|> 就置 True（用户意图被接受），系统开始回答就清零——回答中途用户再插话要重新走接受流程。

七、正确性保障：三层防线（文档 §9）
fast 抄了近路，怎么证明没抄错？

验证者独立性：fast 和单测共享路由代码，但两个参考脚本不共享（路由内联未动）——防止共享代码有 bug 时"同错互证"；
单测 tests/test_route_position_faithful.py：prev_state × state_pred × top1 × config 约 396 种组合逐一对照参考内联逻辑；
--verify_ar（fast:1313+1440）：同一条 wav 用子进程跑 untouched 参考脚本（隔离、无共享状态），然后逐字段 EXACT 对比 chunk_timeline、token ids、token_stream、contract 全字段；prob_log 概率差只做 TOLERANCE 报告。
代价是要把参考脚本再跑一遍（慢 3-8x），所以只抽查用。

八、收尾：contract 产出（fast:1286）
每条 wav 推理完写一份 JSON 档案（tmp + 原子 rename，fast:1304）：

{
  "wav_id": "advbench_00001",
  "info": {
    "response_starts": [{"first_text_pos": 21, "first_text_seconds": 5.04, ...}],  // ← 首字延迟 KPI
    "response_ends": [...],
    "clean_text": "I'm sorry, ...",   // ← 判分只看这个；多轮回答无分隔拼接
    "trace": {"n_total_chunks": 77, "n_text_chunks": 54, "n_listen_chunks": 23, ...}
  },
  "chunk_timeline": [ {idx, phase, state, token_id, sec, audio_group_idx, accepted_user}, ... ]  // 逐拍流水账
}
其中 audio_group_idx 是给 --verify_ar 对账用的"暗记"——每拍恰好吃一组音频、按序吃，专门抓"fast 在错误时机吃音频"这类 bug。

之后 run_eval.py 按 lmms-eval v0.7.3 官方口径（9 子集 7783 条，判分逻辑逐行对齐含官方 bug）送 judge 打分。

九、一张图总结运行逻辑
       ┌──────────────── 每条 wav ────────────────┐
       │                                          │
音频 → precompute_all_audio → [3行/组] × N 组      │
       │                                          │
       ▼                                          │
  ┌─ Phase 1（有真音频）─────────────────────┐    │
  │  控制态 → prefill_control_run:          │    │
  │     草拟 K 拍"都听" → 1次大 forward       │    │
  │     → 逐位验收 ──一致──→ 白赚，继续草拟     │    │
  │               └─不一致→ turn point:      │    │
  │                 重建该拍 → ar_one_chunk   │    │
  │  AR 逐拍: 表态(speak?) → 吐字，每拍仍吃音频│    │
  └──────────────────────────────────────────┘    │
       │ 音频耗尽                                  │
       ▼                                          │
  ┌─ Phase 2（零波形陪跑）───────────────────┐    │
  │  继续每拍吃"安静"，直到说完 & ≥21拍安静     │    │
  └──────────────────────────────────────────┘    │
       │                                          │
       ▼                                          │
  _contract_entry → 原子写盘 → run_eval 打分       │
十、如果要继续深入，建议的研读顺序
先读 fast:1-117 模块 docstring——投机原理和 verify_ar 契约的官方说明；
跟着 generate（fast:426）签名进主循环（fast:1150-1261），把 Phase 1/2 的分支条件画出来；
精读 prefill_control_run（fast:812-1027）——这是全文技术含量最高的部分，重点理解 turn point 处"预测有效但后续丢弃"的论证；
对照 duplex_routing.py:367 把状态机转移表抄一遍；
最后拿一条真实 chunk_timeline（文档 §11 的格式）手动走一遍状态转移，验证自己的理解。
文档末尾的"核实清单"（§14）列出了全部代码锚点，如果你能把 mola_duplex_training 仓库放到本机，我可以带你逐行核实其中任何一处——比如把投机预填充的验收逻辑和参考实现逐行 diff。你想先深入哪一块？


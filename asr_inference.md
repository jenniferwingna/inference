# mim-omni @ ASR_deplex_merge_valid_GPU · duplexasr_infer 深度拆解 · 百科全书版

> 对象：`mim-omni @ ASR_deplex_merge_valid_GPU`（HEAD = `60e39954`，2026-11-03）
> 主角 `duplexasr_infer/` 包（~7800 行）——核心 `chunk_engine.py`（723 行，
> 状态机 + 批引擎）、`model_runner.py`（968 行，前向原语 + CUDA graph + GQA）、
> `token_scaffold.py`（chunk 布局与 Phase 机）、`audio_frontend.py`（音频链路）、
> `ckpt_loader.py`（DCP→bf16 双路径加载）。
>
> **写法**：每个概念三层——**专业**（代码级精确描述，全部带 文件:行号，可逐条跳转核实）
> → **大白话**（类比直译）→ **例子**（本仓实测/文档记录的真实数据）。
> 文末附"核实清单"：本文引用的每一处代码位置。
>
> 与 VoiceBench 拆解（mola_duplex_training）的对照贯穿全文：那是"纯 duplex
> 对话"模型，本仓是 **duplex + chunk-ASR 联合模型**——每拍除了行为决策，
> 还要吐"用户刚说了什么"的流式转录，且**允许反悔上一个拍的转录**。

---

## 目录

1. 全局链路
2. 双路径模型加载（DCP 慢链 / prepared bf16 快链）
3. 模型本体与一拍的 token 布局
4. 流式音频前端（timeline 混音 → mel → WS4 因果塔）
5. 引擎主循环：push_chunk 的三步舞
6. 解码调度：Phase 状态机 + 三层 guardrail
7. rollback：chunk-ASR 的"反悔"机制（本仓灵魂）
8. 提速四板斧：KV cache / CUDA graph / stream_batch / GQA
9. 停止体系：五道闸
10. 验证体系（三层防线）
11. WER 评测闭环
12. 输出产物逐字段
13. 工程层（di_gpu 部署 + vLLM 支线）
14. FAQ
15. 核实清单

---

## 1. 全局链路：这套推理到底在干什么

**专业**：把双工-ASR 联合训练产出的 checkpoint（DCP 格式，两节点 16-rank
或单节点；fp32 master 权重）加载为 VeOmni patched Qwen3-Omni-30B-A3B
thinker（关 talker），对 UDS jsonl 的每条对话逐条执行**逐 240ms chunk 流式
推理**：每拍 prefill 一个 `[chunk_bos + 3 行音频特征]` 的骨架，然后模型
自回归吐出 **chunk-ASR 转录槽（含 rollback 修正）+ agent 槽（1 个行为
token）+ action 槽 + chunk_eos**。零 GT 泄漏：prefill 里没有任何真值
文本（`infer_uds.py` 模块 docstring，`infer_uds.py:8-9`）。

```
OBS DCP ckpt (fp32, 两节点) ──下载/合并/prepare──> bf16 safetensors (~60G)
                                                        │
UDS jsonl ──> timeline 混音 ──> mel ──> audio tower ──┐  │
   (用户音频摆到对话时间轴, 尾部+60s 留白)              ▼  ▼
                                            ModelRunner (层切分 ≤4 卡)
                                                        │
                              ChunkStreamEngine / ChunkBatchEngine
                              每 240ms: prefill 音频块 → decode 槽位序列
                                                        │
                 ┌──────────────────────────────────────┴──────────┐
                 ▼                                                  ▼
        fwd_tokens_uds.txt (逐 token (P)/(D) 日志)        uds_clean.txt (回合级)
                                                                 │
                                              eval_asr_testset.py (wer_i18n)
                                                                 ▼
                                                        wer_summary.json
```

**大白话**：训练出一个"边听边转写边想接话"的模型。推理程序把它装进
GPU，把每通电话切成 240ms 一拍喂给它；每一拍模型要交三份作业——
"用户这拍说了什么"（转录，还能改上一拍的口误）、"我现在该听该说"
（行为）、"要不要调工具"（动作）。

**例子**：di_gpu 冒烟 20 条单卡跑通；test_clean 前 100 条单卡 ≈ 33 分钟
（graph 模式，`di_gpu/README.md:103`）。

---

## 2. 双路径模型加载：慢链与快链

**专业**（`ckpt_loader.py`）——入口三分叉在 `infer_uds.py:138-152`：

1. **慢链（DCP）**：`download_ckpt`（`ckpt_loader.py:169`，moxing
   `copy_parallel`）→ `merge_and_resolve`（`:258`）自动识别两种布局：
   平铺 DCP（`.metadata` + `__{rank}_{i}.distcp`）或**两节点上传布局**
   （`node0/` + `node1/` 各自带 `.metadata`；shard 文件名全局唯一 →
   纯拷贝平铺 + 两份 metadata pickle 取并集，`_merge_metadata:333`）→
   `build_veomni_model`（`:390`，**fp32 建模**——DCP 里存的是 fp32
   master）→ `load_model_into`（`:458`）：单进程 gloo world_size=1
   `DistributedCheckpointer.load`（训练同款链，fused-expert 布局自动处理）。
   全程 ~20 分钟级
2. **快链（prepared）**：`resolve_weights_source`（`:521`）发现目录带
   `PREPARED_BF16.json` 标记（`:511`）→ `build_veomni_model(init_device=
   "meta")`（秒级空壳）+ `load_prepared_into`（`:627`）直接 overlay bf16
   safetensors → **~3 分钟起跑**（`infer_config.py:104-106` 注释原文）
3. **prepare 一次性转换**：`save_prepared_from`（`:537`）把慢链加载完的
   fp32 模型 cast bf16、按 ~8GB/shard 写 safetensors + 标记文件——
   `prepare_ckpt.py` 是独立入口，di_gpu 的 `02_pull_model_ckpt.sh` 自动做

**ops 精确配置**（`ckpt_loader.py:409-421`）：attn=sdpa、MoE=fused_npu
（NPU）/fused_triton（GPU 自动换，`infer_config.py:545-562`
`_normalize_ops_for_device`）、rmsnorm/rotary=npu→liger_kernel、其余
一律 eager——**推理 ops 与训练逐一对应**。

**关键坑**（`infer_uds_plan.md:169-171`）：8 个 duplex 特殊 token 固定
落在 Qwen vocab 的 151693-151700 行（训练 `init_new_token_embeddings=
false`），**嵌入矩阵不扩容**——加载后形状必须与基座完全一致，禁止
resize embedding。

**大白话**：慢链是"把两台机器上拆开存的发动机零件拼好再装车"，
快链是"上次拼好的整发动机直接吊装"。都装好后要验明正身：ops 配置
跟训练一个字母都不能差，否则数值行为就变了。

**例子**：di_gpu 默认 `CKPT_OBS_URL=...duplex-with-asr-debug-480-rb0-
mixed-data-zxm-v1/global_step_5000`，`PREPARED_WEIGHTS_DIR=/cache/
ckpt_infer/duplexasr_all/bf16_weights`（`di_gpu/gpu_env.sh:44-46`）。
加载日志打 `missing=0 unexpected=0`（`di_gpu/README.md:87`）。

---

## 3. 模型本体与一拍的 token 布局

**专业**：主体 Qwen3-Omni-30B-A3B thinker，`disable_talker()` 关语音合成
侧（`ckpt_loader.py:431`）。**1-head**：无 state_head（本 ckpt
`duplex_state_head=False`；若 ckpt 有而 cfg 没开，加载时直接 drop +
警告，`ckpt_loader.py:447-455`、`infer_uds.py:153-158`）。

一拍的布局（训练对齐基准 = `infer_uds_plan.md:31-41`，实现 =
`token_scaffold.py:82-96`）：

```
prefix（全局一次）:
  <|im_start|> system_prompt <|im_end|> <|task_audio2text|> [<|task_chunkasr|>]
                                                    ↑ chunk_asr=true 时追加

每 chunk k（240ms）:
  (P) <|chunk_bos|> <|audio_start|> <|audio|>×3 <|audio_end|>     ← prefill
  (D) <|chunkasr_start|> [R_k] <|rollback_sep|> [F_k] <|chunkasr_end|>  ← 解码
  (D) <|agent_text_start|> {恰好1 token} <|agent_text_end|>
  (D) <|action_start|> {tool tokens 或空} <|action_end|>
  (D) <|chunk_eos|>
```

- **(P)/(D) 分界是本仓的第一不变量**：prefill 只有 prefix + chunk 骨架
  （`token_scaffold.py:9-13` docstring 原文 "zero label leakage"）；
  R_k/F_k/agent/action/eos 全部模型生成
- agent 槽 1 个 token：普通文本 = 回复中；`<|listen|>`（151695 域段）
  = 听；`<|speak|>`/`<|text_break|>`/`<|idle|>`/`<|interrupt|>` = 行为词
  （`chunk_engine.py:79-87` 收集 `_agent_controls`）
- modality_ids：1 标记 `<|audio|>` 占位（`token_scaffold.py:87-96`），
  forward 时音频特征精确 scatter 进去，**数量不符直接 raise**
  （`model_runner.py:217-220`）
- 训练时 R_k 的 input 以 0.75 概率被换成 `<|rollback_blank|>`（防复制
  机制，label 保留真值）——**推理不存在**：asr token 全由模型生成
  （`infer_uds_plan.md:45-48`）

**大白话**：每拍的表格 = 3 格声音 + 一支笔。模型拿到声音后先在"转写
栏"写字（还能划掉上一拍写错的），再在"行为栏"画一个圈（听/说/停），
最后签名交卷。系统指令是贴在整沓表格最前面的总说明。

**例子**：一个含语音的拍，decode 日志长这样（`fwd_tokens_uds.txt` 格式，
`output_writer.py:70-79`）：
`[1.44s-1.68s] <|chunk_bos|>(P) <|audio_start|>(P) <|audio|>x3(P)
<|audio_end|>(P) <|chunkasr_start|>(D) how(D) <|rollback_sep|>(D) do(D)
<|chunkasr_end|>(D) <|agent_text_start|>(D) <|listen|>(D) <|agent_text_end|>(D)
<|action_start|>(D) <|action_end|>(D) <|chunk_eos|>(D)`

---

## 4. 流式音频前端：timeline 混音 → mel → 塔

**专业**（`audio_frontend.py`）——**整条链只读复用训练 builder**，这是
"prefill 与训练比特级一致"的根基：

1. **timeline 混音**（`build_stream:113-122`）：调训练同款
   `_mix_timeline_audio_waveform`（`streaming_reformat/encoded_media`）——
   静音缓冲铺满 `[0, timeline_end)`，每条 user 音频按对话时钟
   `timeline_start_sec` 叠上去；`timeline_end = 最后消息 end + 60s 尾部
   留白`（`cfg.audio_tail_sec`，`infer_config.py:197-199`）；再
   `_pad_waveform_to_chunk_grid` 对齐 240ms 网格
2. **mel**（`_extract_mel:258-270`）：模型目录自带的
   WhisperFeatureExtractor（n_fft=400, hop=160, 128 mel, 10ms/帧），
   `padding=False, truncation=False`——与训练 processor 完全同源
3. **塔编码双模式**（`build_stream:124-131` 分叉）：
   - `streaming`（默认，`infer_config.py:218`）：WS4 因果塔——
     `_encode_streaming:170-256` 用训练同款 `encode_stream`：独立 2s
     smooth_slide buffer（n_window_infer=200 帧）按 80ms 网格排开，
     每 buffer 过一次塔，取 `-emit_offset` 行；**emit_stride = 240/80
     = 3**（`:198`）——每次塔 forward 吐出某一拍的 3 个 slot。
     cadence stamp 用 `check_encoder_cadence(allow_mismatch=True)` 校验，
     不符大声警告但不拦（`:202-214`）
   - `whole`：整段时间轴一次过双向塔（8s 窗，ckpt 原生）——给无
     cadence stamp 的老 ckpt（如 global_step_15000）
4. **leading-slice 契约**（`:132-138`）：训练消费的是塔输出的**前
   `stream_token_count` 行**，推理同样 `rows[:n_emit]`——数量不足直接
   raise
5. **chunk→行映射**（`:146-161`）：`_build_audio_feature_spans` +
   `_annotate_silence_spans` + `_ensure_audio_spans_cover_chunks` 全是
   训练 builder 的原函数——silence-pad chunk 补零行（always-on 麦克风
   语义：没声音的拍也交 3 行"安静"特征）
6. **塔输出的 `.cpu()` 在 `no_grad` 里**（`:231-235` 注释）：否则
   autograd 会把整条塔计算图（~22GB/条）钉在 `AudioStream.rows` 上，
   第 3 条就 OOM——踩过的坑

**大白话**：把一通对话的录音按剧本摆到时间轴上（用户说话的段落摆
到对应时刻，其余全静音，结尾留 1 分钟让模型把话说完），再按训练时
一模一样的"耳朵"（mel + 音频塔）转成特征。耳朵有两种工作模式：
整段听完再总结（whole），或 2 秒滑窗边听边出（streaming，和训练一致）。

**例子**：日志一行（`audio_frontend.py:163-166`）：
`item_id: timeline=88.20s mel=8820f rows=1102 chunks=367 mode=streaming`
——88.2s 时间轴、1102 行特征（88.2×12.5）、367 拍。

---

## 5. 引擎主循环：push_chunk 的三步舞

**专业**（`chunk_engine.py`）：

```
start()（:102-115）: prefill prefix（几十 token），KV 就位
循环 push_chunk(rows)（:117-142）:
  1. PREFILL（_run_chunk:216-229）
     tokens=[chunk_bos, audio_start, audio×3, audio_end]
     → runner.prefill（音频行 scatter，KV 增量追加）
  2. DECODE（:231-251）：phase = initial_phase()
     循环 ≤128 步（:236 安全帽）：
       token, forced = _next_token(logits, phase, ...)  → §6
       result.decoded.append((phase, token))
       phase = sc.transition(phase, token)             → §6
       若 phase ∈ {CHUNK_EOS, DONE}: break
       logits = runner.decode_step(token, cache, past)  → KV+1
  3. PARSE（_parse_chunk:403-446）→ §7 的 rollback 合并
post_chunk（:144-186）→ 停止判定（§9）
```

**大白话**：每拍三步——递耳朵、听回答、记账。递耳朵是一次小
prefill；听回答是逐 token 的贪心解码，最多 128 步；记账把这一拍
的转写按 rollback 规则并进总账，并判断"这通电话要不要挂"。

**例子**：每拍日志（`chunk_engine.py:253-258`）：
`chunk 42: prefill 6tok 147ms, decode 8st 29.5ms, phase=chunk_eos,
past=3181`——prefill 6 个 token（bos+audio_start+3×audio+audio_end）
147ms，解码 8 步（本拍转写 2 词 + 结构 token）每步 29.5ms。

---

## 6. 解码调度：Phase 状态机 + 三层 guardrail

**专业**：Phase 枚举（`token_scaffold.py:24-38`）12 相：
`chunkasr_start → rollback → frontier → chunkasr_end → agent_start →
agent → agent_end → action_start → action → action_end → chunk_eos → done`。

转移表 `Scaffold.transition`（`:125-185`）纯函数：结构 token 驱动相变，
普通文本 token 停在原相。自由度设计（两层模式，`infer_config.py:240`）：

- **free（默认）**：贪心解码 + guardrail **事后裁剪**——状态机只用来
  *解析*（phase 名作为 (D) token 的标注）和算预算
- **sm**：硬约束——采样前就用 `sm_allowed_ids`（`:187-209`）白名单
  拦截（老 inference_streaming 状态机的移植扩展，兜底用）

**guardrail 三层**（`_next_token:291-334`，"预算耗尽→强制结构 token"）：

1. **设备侧 phase mask（C7，默认开，`chunk_engine.py:336-368`）**：
   ban-list（im_start/im_end/chunk_bos/pad-0，`token_scaffold.py:211-219`）
   + 本相 off-schedule 的结构 token，烘成 vocab 布尔 mask 在 GPU 上
   `masked_fill`——**top-64 候选全部合法**，host 循环只管预算（旧路径
   64 个候选可能全是非法 token，白烧一步）
2. **预算帽**（`phase_budget:102-123`）：rollback/frontier 各 ≤24、
   agent 槽 ≤8、action ≤8、连接相 ≤2（训练把结构 token 挨着放，2 已
   宽松）——超了强制 `_forced_structure_token`（`:379-400`，如 frontier
   超额 → 强制 `<|chunkasr_end|>`）
3. **结构调度**：`_phase_structures`（`:264-289`）给出每相允许出现的
   结构 token；`_is_structural`（`:370-377`）8 个结构 token 名单之外
   都算普通文本

**选 token 的实际路径**（`_next_token:309-334`）：`logits.float()` →
（mask）→ `torch.topk(work, 64)` → **只同步 64 个 id 到 host** → 逐个
挑第一个合法的。全词表 argsort+tolist 每步都会卡住循环（`:305-308`
注释）。

**大白话**：模型自由发挥，但有三道裁判：第一道在 GPU 上就把"这个
阶段不该出现的动作词"屏蔽掉；第二道数着"这一栏最多写 24 个词"；
第三道盯着"该收尾了就收尾"。裁判只裁不引导——模型自己想说什么
还是什么（对比 mola 的 v2：那边是硬状态机路线，这边是自由+护栏）。

**例子**：guard 事件计数进 rtf_stats（`infer_uds.py:311`）；A/B 对齐
协议要求 `guard_events` 总量不高于基线（`di_gpu/OPTIMIZATION_PLAN.md:86`）
——它是"阶段机行为哨兵"。

---

## 7. rollback：chunk-ASR 的"反悔"机制（本仓灵魂）

**专业**：训练几何（rollback=1，`infer_uds_plan.md:37-38,54-55`）：

```
chunk k 的 asr 槽 = [R_k] <|rollback_sep|> [F_k]
  R_k = 对 chunk k-1 的 frontier 的修正版（重发）
  F_k = chunk k 自己的新增词
```

推理合并规则（`_parse_chunk:424-446`）：

```python
# rb1:
if rb and self.finalized_asr:
    overlap = len(rb)
    self.finalized_asr = self.finalized_asr[:-overlap] + list(rb)  # 覆盖尾部
self.finalized_asr.extend(fr)                                       # 追加新词
# rb0（无 rollback 的 ckpt）: 纯增量 extend，
#   且首词落在 chunkasr_start 相要专门捞回（:437-444，d75532fe 修的丢词）
```

即：**最终转录流 finalized_asr 中，chunk k 的先锋 F_k 会被 chunk k+1
的修正区 R_{k+1} 覆盖尾部**——修正延迟恰为一拍。每个 ChunkResult 记
`finalized_len`（`:44-45`）= 合并后的流长度快照，供回合切片（§12）。

为什么需要：240ms 只够说 2-4 个 token（`infer_config.py:241-243`），
词跨界处先锋转录必然有"猜错尾巴"——下一拍听到了更多音频，模型
重发修正版。这是**用一个 chunk 延迟换转录精度**的流式 ASR 设计。

**d75532fe / 9746fb1b 两个真实翻车**（都是这个机制的 bug 修复）：
- rb0 模式：decode 循环把 token 记在**转移前**的相——每拍第一个
  内容词落在 `chunkasr_start` 相被丢 → 专门收集（`:437-444`）
- duplex overlap：agent 已开口回复时 ASR 词还在进（转录滞后于回复），
  回合切片缝隙里丢词 → 回合 ASR 窗口扩到下一 span 开前
  （`output_writer.py:214-224`）

**大白话**：速记员每 0.24 秒交一次速记，但允许下一秒"刚才那几个字
应该是……"地划掉重写。总账只留最后一次说法。漏掉这个合并规则的
话，每拍的话都记两遍——WER 直接爆炸 30 倍。

**例子**：`di_gpu/README.md:79` 的定案警告：
`CHUNK_ASR_ROLLBACK=1`——"模型自发输出'重发+rollback_sep+新增'，rb1
才正确 merge；**漏了 WER 虚高 30 倍（32.8% vs 1.56%）**"。

---

## 8. 提速四板斧

### 8.1 KV cache 增量解码（对比参考脚本的浪费）

老参考脚本每步 `use_cache=False` 全量重前向；本仓默认
`INFER_USE_KV_CACHE=1`（`infer_config.py:287`），`--no-kv-cache` 留作
V4 parity 的参考模式（数量级慢，`:285-286` 注释）。engine 里对应
`_full_ref_forward_replay`（`chunk_engine.py:449-458`）。

### 8.2 CUDA graph（B=1 单 token 图）

**专业**（`model_runner.py:766-869` `_setup_decode_graph`）：eager 解码
每步 ~500 个 host dispatch op（CPU 争抢时 450ms wall vs 11ms 真算力，
`:768-771` 注释）。捕获单 token 步为图：静态 buffer `in(1,1,H)` /
`pos(int64[1])` / `out(V)`，mask 和 rotary select **图内**从 buffer 构造
——replay = 拷输入 + `pos.fill_(p)` + `replay()`（`:919-924`）。

四个踩坑全写在代码里：
- **warmup 必须侧流** + 练尾部 slot（M-8..M-6）不碰真 KV 区（`:837-846`）
- **warmup 会推进 StaticCache 的 cumulative_length**——不 reset 的话
  首个真 prefill 从 slot 3 写起而 mask 当它从 0 开始（`:852-856`，
  "E0's trap"）
- **跨 item 复用**：图绑定的 StaticCache 是 runner 级共享资源，每条
  item 前 `runner.reset_kv_cache()`（`infer_uds.py:230`，`model_runner.py:
  267-272`）——这是"跨 item OOM 修复"（cf5bef90）的一半
- **超预算逃生**：`past+1 >= 16384`（`INFER_GRAPH_MAX_LEN`，`:99`）时
  同一 StaticCache 上回退 eager 单步（`_graph_step_token:873-882`）——
  KV 流不断，只是慢

**效果**：decode 29.5ms/步（eager 60ms 且 CPU 争抢放大 5-8 倍），
RTF 0.50 vs eager 6.21（`di_gpu/README.md:96-102`）。

### 8.3 E2 stream_batch：token 级 lockstep 批处理

**专业**（`chunk_engine.py:473-719` + `model_runner.py:348-762`）——
设计文档 `di_gpu/OPTIMIZATION_PLAN.md`（S2）：

- **ItemStepper**（`:473-591`）：把单条 item 的 chunk 流水线改成步进
  机（state: idle→prefix→inputs→decode→finished）。**复用**
  ChunkStreamEngine 的全部单条逻辑（`_next_token`/`_parse_chunk`/
  `post_chunk`/finalized_asr）——"单样本行为 by construction 保持"
  （`:476-481` docstring）。chunk 的 prefill token 也逐个喂（与 graph
  路径的 split prefill 数值路径一致，`:467-470` 注释）
- **ChunkBatchEngine.run**（`:642-694`）：B 条 item 共享一个
  BatchSession——**每 wall-clock 步所有活跃 slot 恰好各写一个 KV
  位置**（uniform-KV-length，`:349-354` 注释）。`past` 是全局写指针，
  `real_len[b]` 是各 slot 真实 token 数；完成的 slot 继续写 pad
  token 的 KV，由 per-slot 可见性 mask 遮住（`model_runner.py:930-942`
  BatchSession docstring）。mask 构造 `keys < real_len`（`:738-741`）
- **prefix 批 prefill**：所有 item 前缀等长（同一 system prompt）→
  一次 `prefill_batch`（`:647-657`；`model_runner.py:579-650`）
- 批内 audio 占比均摊 process_sec（批内单条 wall 时间无意义，
  `infer_uds.py:26-33` docstring）；结果按**原输入序**回写三 writer
  ——输出文件与 graph 模式逐行可 diff（A/B 判据）
- **E3（19925308）**：`enable_batch_graph`（`model_runner.py:388-409`）
  per-B 懒捕获批解码图。图私有内存池装着所有层的 KV-repeat 工作区
  （B×H×M×hd×2×layers，B=8/M=16384 约 100GB）→ `INFER_BATCH_GRAPH_
  MAX_LEN` 默认 8192 封顶，超界同 cache 回退 eager。warmup 必须
  no_grad（否则 repeat_kv 物化 ~54GB 第一波就 OOM，`:536-539` 注释）；
  capture 前 `empty_cache()` 释放 warmup 的工作区（两份并存会 OOM，
  `:548-551`）

### 8.4 E4 GQA fast path（a4b25ebd）

**专业**（`model_runner.py:411-460` `_install_gqa_sdpa`）：monkey-patch
`ALL_ATTENTION_FUNCTIONS["sdpa"]`——**q_len==1 的解码步**直连
`F.scaled_dot_product_attention(enable_gqa=True, attn_mask=...)`，跳过
transformers 包装器的 repeat_kv 物化（GQA 32 组时 B=8/M=8192 每层
2×537MB × 48 层 ≈ 31ms/步，占整步三分之一；bench ~50us/层 vs ~650us）。
q_len>1 的 prefill 走原 wrapper——**prefill 数值与 E2 逐位一致**，
只有解码 kernel 变（广播==物化，数学等价，bf16 ULP 漂移属预期物理
行为，`:424-431` 注释）。

配套坑：E0 的 B=1 图 body 在 gqa 模式改用 **bool mask**——additive
-inf float mask 会走另一个 sdpa kernel，偶发 NaN → MoE router index
assert（第 ~24 条 item 炸，`:807-812` 注释）。幂等守卫防双安装
（`:436-438`）；必须在任何图捕获**之前**武装（`__init__:49-50`）。

**大白话（四板斧总览）**：
1. KV cache：别每步从头背整句话；
2. CUDA graph：把 500 步的手工操作录成磁带一次播放；
3. stream_batch：八通电话一起接，一个 GPU 步同时推进八条线的同一
   个字位置；
4. GQA：注意力里"8 把钥匙开 32 把锁"，别把钥匙复印 32 份，直接
   广播着开。

**例子**：提速演进（`di_gpu/README.md` + OPTIMIZATION_PLAN）：
eager+CPU 争抢 RTF 6.21 → eager 干净 4.02 → graph+rb1 **0.50**；
E1（FA2）预期 5-15%、E2（batch）预期 2-5×，主判据输出 100% 一致
（diff 不静默放宽，`OPTIMIZATION_PLAN.md:33-37`）。

---

## 9. 停止体系：五道闸

**专业**（`chunk_engine.py:144-186` post_chunk + `infer_uds.py:240-262`）：

| 闸 | 位置 | 触发条件 | 语义 |
|---|---|---|---|
| ① turn_over | `chunk_engine.py:165-168` | 音频耗尽 && (text_break 或 listen 且已见过回复) | 语义终点：模型自己挂电话 |
| ② drain guard | `:171-180` | 耗尽后拍数 ≥ max_drain_chunks | 对"不懂语义"的模型的客观硬帽（默认 0=关） |
| ③ trailing-idle | `:181-183` | 耗尽后连续 60 拍无回复 | 语义守卫（agent 槽空/控制 token 都算 idle，`:157-164`） |
| ④ silent-tail | `infer_uds.py:246-259` | **音频未耗尽**时连续 15 拍纯静音无回复 | 长静音段/60s 尾巴不逐拍白解码（C3） |
| ⑤ kv budget | `infer_uds.py:240-245` / `chunk_engine.py:667-671` | past 逼近图/cache 上限 | 越界保护，正常防越界非错误 |

外加 `max_chunks=4096`（`:184-186`）绝对帽、每拍 decode ≤128 步
（`:236`）。

**大白话**：挂电话有五种理由——对方说完了（语义）、对方太久不吭声、
这段本来就是留白静音、内存要爆了、聊了 4096 拍封顶。任何一个成立
就收线，**已记的账全部保留**（`stop:200-210` 与 `finish` 的区别：
stop 不再解码但结果完整）。

**例子**：`di_gpu/README.md:112-113`：日志见 `kv budget`/`silent-tail
guard` stop 属正常防越界，非错误。

---

## 10. 验证体系：三层防线

**专业**：

1. **V3 prefill parity**（`tests/test_prefill_parity.py`，CPU 可跑、
   不需要 ckpt）：同一条 UDS 样本过**训练**数据管线
   （ObsStreamingSFTDataset，生产 chunk_asr 配置）vs 推理 Scaffold，
   三断言：prefix 逐 token 相等（`:59-67`）、每个 chunk 的
   `[chunk_bos..audio_end]` 段逐 token 相等（`:69-100`）、**推理
   prefill 不含任何槽内 GT token**（零泄漏，`:102-113`）
2. **V4 增量=全量**：`--no-kv-cache` 参考模式（`chunk_engine.py:449-458`
   `_full_ref_forward_replay` 每步全量重前向）——KV 路径与全量
   argmax 一致性抽查（计划 `infer_uds_plan.md:348-349`；bf16 归约
   顺序微差属预期）
3. **A/B 输出对齐协议**（`di_gpu/OPTIMIZATION_PLAN.md:80-87`）：
   E0/E1/E2 三实验同 30 条逐字节同输入，`fwd_tokens.txt` 逐条逐
   token 比对 + `uds_clean.txt` 逐行 diff，**目标 100% 一致**；diff
   写入明细文件人工裁决（浮点 kernel 差异 vs 实现错误），**不静默
   放宽**；`guard_events` 作为行为哨兵不得高于基线
4. mock 批测试（CPU 可复跑，`OPTIMIZATION_PLAN.md:159-162`）：
   turn_over 路径、KV 全同步不变量（past 每拍恰 +1）、rb1 rollback
   合并几何、kv-budget 强停、silent-tail、原序回写

**大白话**：抄近路的每一步都有对账——prefill 和训练的原版出题
方式逐字对（防题都没读对）、带 KV 的快速算法和最笨的算法对（防
算错）、优化前后输出对（防快了但变了）。对不上必须人工裁决写
报告，不许装没看见。

---

## 11. WER 评测闭环

**专业**（C1+C2，6169c0de）：

```
外部 ASR 测试集 (test_clean/test_meeting_test, obs jsonl+tar)
  → convert_asr_testset_to_uds.py   ← 转成本引擎的 UDS 格式
  → duplexasr_infer 推理 → uds_clean.txt
  → eval_asr_testset.py             ← 解析 clean 输出 + 权威 ref
  → wer_summary.json
```

- **ref 双源**：UDS jsonl 的 `transcript_text` 为权威（`eval_asr_testset.py:
  63-80` load_refs）；clean 输出的 GT 列只是 fallback（`:102-104`）
- **pred** = 该 item 全部 chunk-asr turn 拼接（`:99`）——finalized_asr
  已含 rollback 修正，无需去重
- **语言感知 WER**（`wer_i18n.py`）：WER_MODE=auto——含 Lo 字符（中文）
  走字符级 CER，纯英文走词级 WER，**与纯 ASR 链 eval125 结果直接可比**
  （`eval_asr_testset.py:14-17`）
- **WER 后端 bit 级 vendored**（`wer_backend.py:1-10`）：`_norm_for_wer`/
  `tokenize_for_wer`/`edit_distance_stats` 从 `wgw_duplexwithasr_910B`
  分支的 `chunk_asr/infer_chunk_asr_ckpt.py` 逐字节拷贝（本分支没有
  那个目录，但指标必须逐位一致）
- **meta 可溯源**（4ab5a5f3）：ckpt 来源、测试集身份/规模、产物清单
  写入 `wer_summary.json` 的 meta 节（`eval_asr_testset.py:123-135`），
  全部可选参数不破坏既有字段

**大白话**：拿标准 ASR 考卷（LibriSpeech test_clean 之类）转成自家
格式考这个双工模型，评分器和老 ASR 评测链一模一样（连代码都是原样
搬来的），分数能和历史基线并排看。

**例子**：`wer_summary.json` 结构：`{meta, wer_mode_requested, note,
n_scored, n_no_ref, items[], overall_by_mode}`，`overall_by_mode` 按
模式聚合 `(S+D+I)/N`（token 加权，`eval_asr_testset.py:145-153`）。

---

## 12. 输出产物逐字段

**专业**（`output_writer.py`）——三件套：

**① fwd_tokens_uds.txt（逐 token 日志）**：每 item 头部一行 prefix
全 (P)，然后每拍一行 `[k*0.24-(k+1)*0.24]` + 压缩游程的 prefill
（`<|audio|>x3(P)`，`_compress_runs:265-273`）+ 逐 token (D)（带相名
隐含在顺序里）；尾部 `# rtf:` 统计行（`:65-85`）。

**② uds_clean.txt（回合级）**：`build_turns`（`:132-249`）切回合：
- agent 回合 = 连续"agent 槽是回复文本"的拍段（`:161-171` runs）
- user 回合 = 两个 agent 回合之间有真实音频的拍段（`:188-193`）
- **turn ASR = finalized_asr 的切片** `[前拍 finalized_len, 本拍
  finalized_len]`（`_chunk_asr_span:202-212`）——天然带 rollback 修正
- **overlap 扩窗**：ASR 滞后于回复时，user 回合窗口扩到下一 span
  开前，防词落缝隙（`:214-224`，9746fb1b）
- GT 按消息时间窗与 chunk 窗重叠匹配（`_gt_for:174-184`）
- 行格式 `[t0-t1] [chunk-asr]: (User)pred <=> (GT)ref`，最后按
  start_sec 排序穿插（`:248`）

**③ rtf_stats.json**：逐 item `{id, audio_sec, process_sec, rtf,
n_chunks, chunk_ms_p50/p95, guard_events}`（`infer_uds.py:303-312`）。

**大白话**：一本流水账（每个 token 谁给的）、一份誊清稿（按回合、
和 GT 对照）、一张体检表（每条耗时和裁判干预次数）。

---

## 13. 工程层

### 13.1 di_gpu 一条龙（60e39954）

8 卡 GPU 节点部署：`01_setup_env`（依赖自检不擅自装）→ `02_pull_
model_ckpt`（基座 + DCP → prepared bf16，371GB 下载 1-3h）→ `03_
fetch_data`（测试集 → UDS）→ `04_smoke_infer`（20 条单卡冒烟）→
`05_full_infer_8gpu`（**8 分片 × 1 卡数据并行**，`NUM_SHARDS=8
INFER_NPUS=1`；不推荐单实例 8 卡模型并行——cuda 下未验证）→
`06_eval_wer`（合并分片 + 全集 WER）→ `07_run_all`。全 env 可覆盖，
唯一配置入口 `gpu_env.sh`（`di_gpu/README.md:37-48`）。

**分片约定**：`INFER_SHARD/INFER_NUM_SHARDS` 按行号取模划分数据集，
limit 按 shard 生效（eval125 约定，`uds_dataset.py:96-101`）。

**参考基线**（本机 GPU4 单卡，`README.md:96-103`）：
graph+rb1 RTF **0.50** / eager 4.02 / eager+CPU 争抢 6.21；
test_clean 100 条单卡 33 分钟，全集按条数 ÷8 卡线性外推。

**常见坑**（`README.md:106-113`，全踩过）：rollback 漏设 WER 虚高
30 倍；pypi 假 moxing 要内网源；NPU ops 名在 cuda 自动换 kernel；
同节点共跑重任务 eager 放大 5-8 倍（graph 免疫）；换 ckpt 忘换
PREPARED_WEIGHTS_DIR 会混权重；长音频三层防护触发属正常。

### 13.2 vLLM 支线（`duplexasr_infer/vllm/`）

另一条推理路径（NPU 上验证）：**一个持久 vLLM LLM（TP=2）跑全程**，
每 chunk 是一个独立 request，`prompt_embeds` = 该 item 增长上下文的
切片——vLLM 的 APC 按 KV block hash embeds 字节，chunk k+1 **复用
chunk k 的全部 KV** 只 prefill 新尾巴（`vllm_engine.py:9-17` docstring）。
音频塔跑在 vLLM 外面（`NpuAudioTower`）——vLLM 原生路径会整段重编码
（O(n²) 塔工作量）且表达不了 240ms/2s 窗 cadence（`:12-14`）。Phase
机做成 logits processor 注入（`phase_processor.py`）。权重需先过
`convert_weights_for_vllm.py`：fused experts（`gate_up_proj [E,2i,H]`）
拆回 per-expert HF 布局——**split 顺序用基座 ckpt 余弦验证过**
（cos=0.9999，docstring）。

### 13.3 cloud/ 支线（27e898d1）

NPU 平台（ModelArts 云上）的 01-07 脚本族，仿 eval125 两件套——与
di_gpu 自包含互不依赖。

---

## 14. FAQ

**Q: 这套和 VoiceBench 那套（mola inference_streaming_fast.py）什么关系？**
同一血统的两个分支：模型同源（Qwen3-Omni thinker + VeOmni），任务
不同——那边每拍只出"状态+回复"（audio2text），这边每拍多出
**chunk-ASR 转录槽 + rollback 修正**。提速思路也不同：那边靠"控制
态投机预填充"（草拟 64 拍一次 forward），这边靠 KV cache + CUDA
graph + lockstep 批 + GQA——本仓的 decode 是**结构化的短序列**
（每拍 ≤128 步、typical ~10 步），投机收益小；批处理收益大。

**Q: prefill 为什么要逐 token 喂（graph 模式）？**
NPU 上 q_len>1 段落在 past 之上时 sdpa 的 top-left is_causal 对不齐、
FlashAttentionScore 拒绝 Sq≠Skv 的 mask（`model_runner.py:240-243`
注释）→ split prefill。GPU 上可 opt-in `use_batched_prefill`（显式
bottom-right 因果 mask，`:308-322`）。

**Q: rollback 漏了为什么会 30 倍 WER？**
模型每拍自发"重发上一拍尾巴 + 新词"；不按 rb1 覆盖合并的话重发
内容全部记重 → 大量插入错误。1.56% → 32.8%（`di_gpu/README.md:79`）。

**Q: stream_batch 里完成的 item 怎么办？**
继续写 pad token 的 KV（保持 lockstep 形状），per-slot 可见性 mask
把 pad 的 key 全遮住——真实注意力等价于提前退场（`model_runner.py:
707-741`）。

**Q: 为什么 batch 前缀必须等长？**
`prefill_batch` 要求（`model_runner.py:593-594`）；system prompt 全局
共享所以天然等长（`chunk_engine.py:648-650`）。

**Q: GQA patch 会影响 prefill 数值吗？**
不会——dispatch 守卫只放 q_len==1 进快路，prefill 走原 wrapper
（`model_runner.py:443-446,452-455`）。解码步数学等价但 kernel 不同，
bf16 ULP 漂移按"ASR WER + 无 runaway"判（a4b25ebd 提交说明）。

**Q: silent-tail guard 会不会截断模型回复？**
不会——agent 槽有回复文本就清零计数（`infer_uds.py:249-253`）；
语音停顿（还在说话的间歇）不触发。触发的是纯静音 + 模型没话说的段。

**Q: max_drain_chunks 为什么默认 0（关）？**
语义终点（text_break/listen）和 trailing-idle 已足够；drain 是对
"未训好的模型"的客观保险丝（`infer_config.py:270-275` 注释）。

**Q: 怎么换测试对象（ckpt）？**
`gpu_env.sh` 三件套：`CKPT_OBS_URL` + `PREPARED_WEIGHTS_DIR`（新目录
防混权重）+ `CHUNK_ASR_DELAY_MS`（跟 ckpt 名走的训练记录值，
`di_gpu/README.md:62-66`）。

---

## 15. 核实清单（本文引用的全部代码位置）

| 引用 | 位置 | 内容 |
|---|---|---|
| 零泄漏声明 | infer_uds.py:8-9 | docstring |
| 加载三分叉 | infer_uds.py:138-152 | random/prepared/DCP |
| geometry contract 回显 | infer_uds.py:192-206 | C5 防静默错配 |
| per-item 主循环 + 三 guard | infer_uds.py:218-262 | kv budget/silent-tail/finish |
| stream_batch 驱动 | infer_uds.py:25-118 | 原序回写 + 占比均摊 |
| 配置总表 | infer_config.py:65-384 | 全字段（默认 < env < CLI，:12） |
| ops 设备自动换 | infer_config.py:545-562 | fused_npu→fused_triton 等 |
| DCP 布局合并 | ckpt_loader.py:248-388 | node0+node1 / metadata 并集 |
| fp32 建模 + state_head drop | ckpt_loader.py:390-456 | 训练同款链 |
| prepared 标记/保存/加载 | ckpt_loader.py:511,537,627 | 快路径 ~3min |
| token 契约（151693-151700） | infer_uds_plan.md:169-171 | 禁 resize embedding |
| Phase 枚举 + 转移表 | token_scaffold.py:24-38,125-185 | 12 相状态机 |
| prefix/chunk prefill | token_scaffold.py:70-96 | (P) 契约 |
| 预算/白名单/ban-list | token_scaffold.py:102-123,187-219 | guardrail 参数 |
| 引擎三步舞 | chunk_engine.py:117-261 | push_chunk/_run_chunk |
| post_chunk 五闸 | chunk_engine.py:144-186 | turn_over/drain/idle/max_chunks |
| _next_token + 设备侧 mask | chunk_engine.py:291-368 | topk=64 + C7 phase mask |
| rollback 合并（rb1/rb0） | chunk_engine.py:424-446 | 本仓灵魂；d75532fe |
| 参考模式（无 KV） | chunk_engine.py:449-458 | V4 用 |
| ItemStepper | chunk_engine.py:473-591 | 步进机（复用单条全部逻辑） |
| ChunkBatchEngine | chunk_engine.py:594-719 | lockstep + guards |
| 层切分放置 | model_runner.py:63-127 | ≤4 卡 + arange/rot 缓存 |
| forward + audio scatter | model_runner.py:185-264 | 数量严格校验 :217-220 |
| split prefill / batched prefill | model_runner.py:274-333 | NPU mask 限制 :240-243 |
| B=1 CUDA graph | model_runner.py:766-924 | warmup 侧流/reset 陷阱/超界回退 |
| E3 批图 | model_runner.py:388-577 | no_grad OOM/empty_cache/MAX_LEN |
| E4 GQA patch | model_runner.py:411-460 | dispatch 守卫 + 幂等 |
| prefill_batch 可见性 mask | model_runner.py:579-650 | keys < real+q+1 |
| 批解码 eager/graph | model_runner.py:652-762 | pad 行 + 同 cache 回退 |
| BatchSession | model_runner.py:927-947 | past/real_len/pad_id=0 |
| timeline 混音 + leading slice | audio_frontend.py:101-167 | 训练 builder 只读复用 |
| WS4 因果塔编码 | audio_frontend.py:170-256 | emit_stride=3 / cadence check |
| mel 提取 | audio_frontend.py:258-270 | padding/truncation=False |
| no_grad .cpu() 防 OOM | audio_frontend.py:231-235 | ~22GB/条 的坑 |
| UDS 解析 + shard 约定 | uds_dataset.py:51-149 | eval125 取模分片 |
| 回合切分 + overlap 扩窗 | output_writer.py:132-249 | 9746fb1b |
| V3 parity 三断言 | tests/test_prefill_parity.py:59-113 | prefix/逐 chunk/零泄漏 |
| WER 后端 vendored | wer_backend.py:1-93 | 逐字节拷贝来源声明 |
| 评测入口 + meta | eval_asr_testset.py:83-168 | uds 权威 ref + meta 节 |
| A/B 对齐协议 | di_gpu/OPTIMIZATION_PLAN.md:33-37,80-87 | 100% 一致不静默放宽 |
| E2-E4 提交说明 | git: c21f8916/19925308/a4b25ebd | 设计动机与坑全记录 |
| RTF/性能基线 | di_gpu/README.md:79,96-113 | 0.50 / 29.5ms / 30 倍 WER |
| vLLM 架构 | vllm/vllm_engine.py:9-24 | APC 前缀复用 + 外置塔 |
| fused experts 拆分 | vllm/convert_weights_for_vllm.py:1-14 | cos=0.9999 验证 |

> 所有"大白话"和"例子"是对上述代码与仓内文档的翻译；性能数字
> （RTF 0.50、29.5ms/步、WER 1.56% vs 32.8%、~50us vs ~650us/层、
> 100GB 工作区）均出自 `di_gpu/README.md`、`OPTIMIZATION_PLAN.md` 与
> 代码注释，未做独立复测。

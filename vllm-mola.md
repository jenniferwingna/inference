# vllm-mola @ demo_2heads_di · duplex 推理效率深度拆解

> 对象：`vllm-mola @ demo_2heads_di`（HEAD = `7bcccbc`）。
> 你的工作分支 `c00929682/air-bench-di` 正是从这个提交分出（评测层 `di/`
> 有增删，`vllm_omni/` 引擎栈两侧一致）——即**评测跑的就是这套代码**。
>
> 主角：`vllm_omni/experimental/fullduplex/qwen3_omni/worker.py`（1556 行，
> duplex 会话的 worker 侧核心）、`streaming_audio.py`（1339 行，流式音频
> 前端 + 跨 session 塔批）、`tower_cuda_graph.py`（288 行，塔整图捕获）、
> `openai/runtime_bridge.py`（engine clock）、`core/sched/omni_ar_scheduler.py`
> （2-head 调度）。
>
> **写法**：专业（代码级，带 文件:行号）→ 大白话 → 例子（提交史里的
> 实测数字）。第 5 章给**效率演进时间线**（全部来自 git 提交说明的实测），
> 第 8 章给剩余优化空间判断。文末附核实清单。

---

## 目录

1. 全局链路：评测怎么调用这个 vLLM
2. 部署拓扑：stage0 = 7×TP1 thinker，stage1/2 = 死重量
3. 请求形态与调度：一个 chunk 一次 forward
4. 效率优化栈逐件拆解
5. 效率演进时间线（实测数字汇总）
6. 2-heads 专属机制（本分支的存在理由）
7. 配置逐参数解释（DI 主推 yaml）
8. 当前效率状态与剩余优化空间
9. FAQ
10. 核实清单

---

## 1. 全局链路：评测怎么调用这个 vLLM

**专业**：DI 评测走 **serve 模式**（非 offline batch）：`run_raw_out.sh`
先 ensure_service（按 yaml 起 vLLM duplex 服务），第三步用
`run_realtime_batch.py --url WS_URL --concurrency 48 --no-realtime-pacing`
把每条样本的 events 回放成 WebSocket duplex 会话（`di/run_raw_out.sh:593`）。
客户端并发模型：`asyncio.Semaphore(concurrency)` + `gather`，48 条会话
并发；每条会话按 chunk 推 pcm_f32le 音频（每 append 恰好 3840 样本
= 240ms @ 16k，`worker.py:984-987`）；`--no-realtime-pacing` 关掉实时
节拍全速打；session 准入带重试（max_sessions 有限，2s×attempt 退避，
`run_realtime_batch.py:89-152`）。

```
run_raw_out.sh (workunit 调度)
  ├─ [1/2] convert: OBS → events（缓存三级查找）
  └─ [3/3] run_realtime_batch.py --concurrency 48 --no-realtime-pacing
              │ asyncio.Semaphore(48) + gather（:89,:211）
              ▼
        WS :8092 → FastAPI serving → runtime_bridge（engine clock 门控）
              ▼
        duplex 控制面 → 每 session 的 chunk append → worker 批处理
              ▼
   stage0 thinker ×7 副本（GPU 0-6，FULL CG）→ 文本增量回传
   stage1/2 talker/code2wav（GPU 7，评测不触发）
```

**大白话**：评测器是个"同时打 48 通电话的机器人"，每通电话 240ms
喂一句耳朵数据，服务器像真实双工助手一样边听边回。不模拟真实节奏
（no-realtime-pacing），能喂多快喂多快——测的是吞吐不是时延。

**例子**：vcb 音频批 conc=48（`run_raw_out.sh:198-199`）；air/dynamic_
superb 同档；t4/t68 anchor 也 48。

---

## 2. 部署拓扑：stage0 = 7×TP1 thinker，stage1/2 = 死重量

**专业**（`qwen3_omni_duplex_thinker_7xtp1_mns32_cg.yaml` + `duplex_pipeline.py`）：

pipeline 三 stage（`duplex_pipeline.py:22-60`）：
- **stage0 = thinker**（LLM_AR，owns_tokenizer，multimodal，输出
  latent→text）——**评测的全部计算都在这里**
- stage1 = talker（LLM_AR，stop_token_ids=[2150]）——TTS 侧
- stage2 = code2wav（LLM_GENERATION，输出 audio）——TTS 侧

部署拓扑（7xtp1 yaml:30-48）：
- **stage0：7 副本 × TP1，GPU 0-6 每副本独占一卡**——7 份完整 thinker
  权重，容量 7×mns32=**224 并发会话**（max_sessions 128 硬顶）
- **stage1/2：GPU 7 单副本，mem util 0.15/0.08，enforce_eager**——yaml
  注释原话"thinker_only 死重量"：评测 ckpt 是 thinker-only 权重，这两
  个 stage 装的是空壳/占位，**只占显存不干活**；mns 仍配 48（对称 7bcccbc）
- 容量账：224 = 7×32；conc=48 只占 21%；max_sessions 128 封顶

为什么 TP1 × 7 而不是 TP4 × 2（演进注释，yaml:11-14）："TP4 每层
all-reduce 固定成本主导，TP1 去掉全部 NCCL"——小模型小 batch 下通信
占比高于算力，7 个独立副本吞吐远高于 2 个 TP4 副本。

**大白话**：一个 8 卡机器，7 张卡各摆一个"完整耳朵+大脑"（thinker），
第 8 张卡摆着基本不用的嘴巴（talker/code2wav 死重量）。48 通电话分散
到 7 个大脑上。不用张量并行是因为"八个人平分一道题"的沟通成本，
比"每个人独立做题"还贵。

**例子**：旧 2×TP4 跑全量 VCB 要 11h25m，7×TP1+CG 只要 3h11m
（e6a3ff1 实测，3.6×）。

---

## 3. 请求形态与调度：一个 chunk 一次 forward

**专业**：duplex 会话没有传统 prefill/decode 两段——**每个 240ms
chunk 就是一次独立 forward 请求**：

- 每 append = 3840 样本（240ms）→ 切 3 个 80ms 子块喂 `StreamingMelPrefix
  Encoder`（holdback=1、running-max re-clamp，`worker.py:960-970`）→
  `StreamingAudioEncoder`（因果 26-token 滑窗、lookahead=3、shift_output，
  `worker.py:898` docstring）→ 产出 **13 token/chunk 的音频特征行**
  （`tower_cuda_graph.py:44-46`：100 帧/chunk、13 token/chunk）
- **per-req token 形状只有 {3, 9, 51} 三种**（yaml:16 形状审计）——
  chunk 骨架 + 本拍转写/行为 token，无 1-token decode
  （2-head 模型下 agent 槽由 state_head 路由强制，不走自回归长解码）
- 这正是 LM 侧 FULL CUDA graph 可行的前提：形状种类少且无 decode
  步 → capture sizes [16..512] 全覆盖，padding 实测仅 **1.12×**（yaml:17）
- `omni_ar_scheduler`（vLLM-omni 的 AR 调度器）逐 chunk 调度，2-head
  ckpt 还要 `QWEN3_OMNI_EMPTY_ACTION_BLOCK=1`（§6.3）

**大白话**：不是"读一整段再生成"的传统 LLM 调用，而是"每 0.24 秒
敲一次门"。每次敲门带的 token 极少（3/9/51 个），所以整个服务像
高频小包转发器——瓶颈不在算力在调度和固定开销。这解释了为什么
优化手段全是"消固定开销"（CUDA graph、合批、缓存）而不是"算得更快"。

**例子**：engine clock 关闭时，每 session 每 chunk 独立触发一次
engine step；mns=48 时一个 batch 最多塞 48 个这样的小 forward。

---

## 4. 效率优化栈逐件拆解

### 4.1 三件套（e6a3ff1，"流式吞吐三件套"，默认启用）

**① engine clock（全局合拍网格，当前默认关闭）**

专业（`runtime_bridge.py:199-227`）：无状态全局时间网格，所有 waiter
睡到下一个 `tick_ms` 整倍数边界——同 tick 内并发 session 的 chunk
append 到达同一调度窗口，**单次 forward 批处理**。仅门控非 final 的
`append_audio_chunk`（turn commit/token append 即时通过）；per-session
线序由 append task 链保序（`native_append_tail`）。
配置：yaml `duplex_session.engine_clock_ms` 或 env
`VLLM_OMNI_DUPLEX_ENGINE_CLOCK_MS`；**tick=0（默认）= 关闭，即事件
驱动即时 append**。

大白话：48 通电话各自想什么时候敲就什么时候敲（现状）；engine
clock 是让所有人按同一个节拍器敲——这样门卫一次收一批。**注意：
DI 主推 yaml 没配这个字段，此优化处于关闭状态**（见 §8）。

**② tower 跨 session 批处理（deferral sink，默认开）**

专业（`streaming_audio.py:935-941` + `worker.py:930-950`）：给每个
session 的 `StreamingAudioEncoder` 挂 `defer_sink`——push/flush 不再
每 token 跑塔，把就绪的 mel buffer 挂到 worker 级
`_deferred_tower_pending` 列表；`preprocess_batch`（engine step 的批
钩子）把**所有 session 的 buffer 收拢，每 step 只发一次 batched tower
forward**。bookkeeping（_emitted/_next_token/_trim）照旧推进——
buffer 与同步路径 bitwise 一致（:935-941 注释）。开关
`VLLM_OMNI_DUPLEX_TOWER_BATCH`，默认 1（`worker.py:99-102`）。

大白话：以前每个 session 每拍都要单独叫醒一次耳朵；现在所有电话的
耳朵数据攒成一摞，一次算完再分发回去。省的是"叫醒成本"（kernel
launch 链）不是"算术本身"。

**③ tower CUDA graph（整图捕获，默认开）**

专业（`tower_cuda_graph.py`）：跨 session 批之后塔每 step 只跑一次，
但一次仍是 ~30 层 × (qkv/attn/out/fc1/fc2) 的 Python dispatch 链
（:1-9）。流式几何完全静态——每 buffer 恒 200 帧 = 26 token——
于是按 **bucket（1/2/4/8/16/32/64 个 buffer）** 整图捕获：
- stock forward 不 capture-safe（`.tolist()`/`.max().item()`/
  `pad_sequence` host sync，:11-13）→ `_forward_fixed`（:218-280）
  对等长 chunk 重推 prologue：split+pad→reshape、conv mask 全真→
  masked_select→view——**bitwise 等价已验证**（:14）
- 唯一残余 sync 在 flash-attn custom op 的 `max_seqlen.item()`——
  特意做成 CPU tensor（capture-legal，kernel 收到的还是 CUDA
  cu_seqlens，:17-23）
- 桶池共享一个 graph_pool（:187-190）；非固定几何/超桶/捕获失败
  → 永久回退 eager（kill-switch `VLLM_OMNI_TOWER_CUDA_GRAPH=0`）

大白话：耳朵的计算也录成磁带。因为每拍的耳朵形状永远一样（26 个
token），可以按"这次有几摞"分 7 档磁带，来几摞放哪档。唯一难点是
原版代码里有几处"停下来看一眼数字"的操作，得手工改写成磁带友好的
等价形式。

**④ tokenizer 缓存**（同提交）：session open 不再每次重解析 vocab
JSON——**~0.5-1s CPU/次，曾占 serving 进程 profile 主导**
（e6a3ff1 提交说明）。

### 4.2 LM 侧 FULL CUDA graph（yaml stage0）

专业（7xtp1 yaml:15-17）：stage0 `enforce_eager: false` +
`compilation_config {level: 0, cudagraph_mode: FULL, capture_sizes
[16..512]}`。FA3 的 `_cudagraph_support=ALWAYS` 走 vLLM 原生 FULL
capture；**level:0 = 不跑 Inductor 编译**——规避 B3 路线 -4.68pp
精度退化（yaml:17）。实测 basic_math 146 题：eager 121 → CG 205
题/min（**1.69×**）。

### 4.3 stage1/2 mns 16→48（7bcccbc，最新 perf）

专业（yaml:40-42 注释）：7×stage0 的会话 chunk 流**全部路由到 GPU7
单副本 stage1/2**——mns=16 上限下 48 并发排队，单样本 elapsed 6×
（18.5→102.6s median），conc 8→48 只 +9%。smoke 单卡实测：mns 16→48，
basic_math 0.581→0.333 s/sample（**1.75×**），单样本 median 60s+ →
14.3s，显存预算不变。**conc=8 时 8<16 不排队，所以旧配置在低并发下
从未暴露**。

大白话：七个前台接了客，全部挤到唯一一间办公室盖章。办公室一次只
放 16 人，排队排到天荒地老。把办公室扩到 48 个位子，队伍消化掉了。

### 4.4 客户端并发 8→48（eb37611）

专业：DI 实跑 t4/t68 发现 conc=8 只占容量 224 的 **3.6%**，整体 GPU
利用率 **13.12%**；钉子样本（长尾）占满 8 槽后吞吐坍缩（mtg_top50
5.45→21.72 s/sample）。48 = 容量 21% / max_sessions 128 顶的 37%，
每副本约 7 会话 << mns 32。shard_policy.yaml 三组（vcb 通配 / t4 /
t68 anchor）+ `run_raw_out.sh get_conc` 同步 48。

---

## 5. 效率演进时间线（全部为提交说明实测）

| 阶段 | 改动 | 实测 | 来源 |
|---|---|---|---|
| 基线 | 2×TP4 topology | 全量 VCB phase1 **11h25m** | e6a3ff1 |
| TP4→TP1 | 7 副本×TP1，去全部 NCCL | TP4 all-reduce 固定成本主导 → 拓扑重排 | yaml:11-14 |
| 三件套 | engine clock + 塔批/塔图 + tokenizer 缓存 | 1×TP1 副本 42→**108 题/min（4.1×）** | e6a3ff1 |
| LM FULL CG | stage0 cudagraph_mode FULL + level 0 | basic_math 121→**205 题/min（1.69×）**；全量 VCB phase1 **3h11m（3.6×）** | yaml:18, e6a3ff1 |
| stage0 扩容 | 2 副本→3×TP2→7×TP1 | 2 副本利用率 <20%（c7ce84a） | c7ce84a, cad0963 |
| stage1/2 mns | 16→48 | 0.581→**0.333 s/sample（1.75×）**；单样本 median 60s+→14.3s | 7bcccbc |
| 客户端并发 | 8→48 | 占容量 3.6%→21%；GPU 利用率 13%→提升 | eb37611 |

**累计**：全量 VCB 11h25m → 3h11m（≈3.6×）；单题速度累计 ~10×
量级（42→205 题/min）。

---

## 6. 2-heads 专属机制（本分支的存在理由）

### 6.1 双头路由采样

专业（`worker.py:1396-1480` `_prepare_duplex_sampling_2head`）：
thinker 的 `state_head`（3 类：0=LISTEN / 1=SPEAK / 2=IDLE）在
agent-slot 位置产出 `state_logits`；采样时按行取
`state_pred = argmax(state_logits[row])`，连同 lm_head top5 一起喂
`route_position(prev_state, graph=...)`（`duplex_routing.py`——与
mim-omni 训练侧同源的状态机路由），得到 `emitted_token, new_state`，
然后 **mask 全词表只留 emitted_token 强制采样**——token 被直接写进
KV，agent 槽的决策完全由双头信号决定，不做自由生成。图
（DuplexTransitionGraph）按 session-static 能力一次性构建缓存
（`worker.py:1234`）。

大白话：普通模型的"该不该说话"靠文字 token 的概率；双头模型有
一个专门的"意图头"直接输出听/说/待机。服务端把意图头的答案盖在
文字头上——嘴只能按头的决定动。

### 6.2 prev_state 会话态

专业：每行按 `(session_id, incarnation)` 找 `_duplex_sessions` 取
`prev_state`（:1433-1440）——路由是状态机，上一拍的状态决定合法
转移。`_duplex_state_routed` 去重键 `(request_id, seq)` 防重复路由
（:1428-1431）。

### 6.3 EMPTY_ACTION_BLOCK（2-heads ckpt 起服必须）

专业（`omni_ar_scheduler.py:410-418`）：1h LISTEN+INTERRUPT 消融系
ckpt 训练时 `<|action_start|><|action_end|>` 相邻且两 token label=-100
（不监督）——自由解码会吐 ~10 个结构 token 污染下一 chunk 上下文。
调度器强制 action 段 `allowed_token_ids=[action_end]` 立即闭合。
这就是所有 yaml 头部要求的 `export QWEN3_OMNI_EMPTY_ACTION_BLOCK=1`。

大白话：这批模型训练时没学过"填动作栏"，会乱涂乱画污染下一题；
直接规定"动作栏必须当场交白卷"。

---

## 7. 配置逐参数解释（7xtp1_mns32_cg.yaml）

| 参数 | 值 | 为什么 |
|---|---|---|
| stage0 devices/replicas/TP | GPU0-6，7×TP1 | 7 份独立 thinker，去 NCCL；容量 224 |
| stage0 enforce_eager | false | 走 FULL CG |
| stage0 cudagraph_mode / sizes | FULL / [16..512] | per-req 只有 {3,9,51} 三种形状、无 decode，FULL 全覆盖 |
| stage0 level | 0 | 不跑 Inductor（B3 -4.68pp 精度退化） |
| stage0 max_num_seqs | 32 | 每副本 batch 上限；7×32=224 |
| stage0 gpu_memory_utilization | 0.90 | 权重+KV+CG 池 |
| stage0 prompt_context_guard | 60000 | 超长上下文防护（DI 已把 max_model_len 提到 131072，c7ce84a 注释：65545 长样本曾 broadcast 崩溃） |
| stage1/2 devices/mem | GPU7，0.15/0.08 | thinker-only 死重量，仅占显存 |
| stage1/2 max_num_seqs | 48 | 7bcccbc：全流量挤单副本，16 排队致 6× 延迟 |
| stage1/2 enforce_eager | true | 无流量，不值得捕图 |
| duplex_session.max_sessions | 128 | 硬顶（容量 224 的 57%） |
| async_chunk / active_stream_window | true / 1 | chunk 异步流水，激活流窗口 1 |
| （未配）engine_clock_ms | 默认 0=关 | **三件套之一当前关闭**，见 §8 |

---

## 8. 当前效率状态与剩余优化空间

**专业判断**（基于配置与提交史交叉）：

1. **engine clock 仍关闭**——三件套里唯一没在 DI yaml 启用的。
   它的适用前提是"多 session 到达时间分散、合批窗口不足"；现在
   conc=48 分 7 副本（每副本 ~7 会话），step 级自然合批可能已够。
   但 eb37611 测得的 13% GPU 利用率是 conc=8 时代的数据，48 并发后
   的实际利用率**没有记录在案**——若仍偏低，engine clock（建议
   tick≈240ms 的整倍数）是最现成的下一步。
2. **stage0 7 卡 + stage1/2 死重 1 卡** = 12.5% 的卡在陪跑。若
   评测长期纯 thinker-only，可探索 stage1/2 合并或按需懒加载，
   换回第 8 卡做第 8 个 thinker 副本（容量 224→256）。
3. **mns 48 是否到位未二次校准**——7bcccbc 的 1.75× 是单卡 smoke
   （B=1 拓扑）数字；7×TP1 生产拓扑下 stage1/2 的最优 mns 可能
   不同（7 副本流量 × 每副本 ~7 会话的到达分布不同）。
4. **吞叶尾部**：eb37611 提到"钉子样本占满槽后吞吐坍缩"——
   workunit 的 LPT 切分 + tail stealing（202254f）是数据侧缓解；
   引擎侧无 per-session 公平队列，长样本仍可长期占 mns 槽。
5. **延迟口径**：FULL CG + 合批优化的是吞吐；交互场景的 TTFS
   （首字延迟）没有专门机制（engine clock 反而会增加固定 0~tick
   的等待）——评测（吞吐）与 demo（时延）的最佳配置**不同**。

**大白话**：这套栈已经把"固定开销"啃得很干净（合批、两张图、
缓存），当前主要矛盾从"单步太慢"转向"排队和空转"——并发到没到
位、八卡用没用满、长尾钉子户。下一个最大杠杆大概率是：先把
conc=48 下的真实 GPU 利用率测出来，再决定开 engine clock 还是
继续加并发。

---

## 9. FAQ

**Q: 评测到底用没用 talker/code2wav？**
没用。评测只要文本（stage0 的 latent→text 输出）；stage1/2 是
死重量——占 GPU7 的 0.15+0.08 显存，enforce_eager，基本无计算。

**Q: 为什么 per-req 形状只有 {3,9,51}？**
chunk 骨架 + 本拍内容的 token 数：3 = 纯静音拍、9 = 带音频拍
（骨架+13 音频行占位？形状审计口径）、51 = 内容最多的拍。关键是
**种类少且固定**，FULL CG 的 padding 只有 1.12×。

**Q: 三件套需要配置吗？**
塔批 + 塔图 + tokenizer 缓存默认启用（env 开关可关）；**engine
clock 默认关闭**，要开需在 yaml `duplex_session.engine_clock_ms`
配 tick。

**Q: 2-heads 和 1-head ckpt 起服差别？**
2-heads 走 state_head 路由强制采样（§6.1），且必须
`QWEN3_OMNI_EMPTY_ACTION_BLOCK=1`；1-head 走 lm_head top1 + 状态机
（`3f7e934` 提供了 1-head ablation serving 支持）。

**Q: 会话超时/清理？**
idle_ttl_s 300、disconnect_grace_s 30、reaper 5s 扫一轮
（base yaml duplex_session 段）——评测客户端断开后 5 分钟内槽位
可复用。

**Q: 和 mim-omni duplexasr_infer（上一份拆解）什么关系？**
同一模型血统的两条推理路：mim-omni 是**自研 PyTorch 引擎**（KV
cache + CUDA graph + lockstep 批，单进程层切分）；vllm-mola 是
**vLLM 化的 serving 栈**（多副本 + vLLM 调度器 + WS 会话）。2-head
路由（route_position/状态机）同源；音频塔的"跨 session 批 + 固定
形状整图"与 mim-omni 的"批 decode 图"是同一思想在两个栈里的实现。

---

## 10. 核实清单

| 引用 | 位置 | 内容 |
|---|---|---|
| 评测调用链 | di/run_raw_out.sh:593,198-199 | run_realtime_batch + conc=48 |
| 客户端并发模型 | run_realtime_batch.py:89-152,211 | Semaphore+gather+准入重试 |
| pipeline 三 stage 语义 | model_executor/models/qwen3_omni/duplex_pipeline.py:22-60 | thinker/talker/code2wav |
| 7×TP1 拓扑 + 演进注释 | deploy/qwen3_omni_duplex_thinker_7xtp1_mns32_cg.yaml:11-48 | 全部数字出处 |
| stage1/2 mns 16→48 | 同 yaml:40-42 + git 7bcccbc | 1.75×、6× 排队 |
| 塔批默认开 | worker.py:99-102 | VLLM_OMNI_DUPLEX_TOWER_BATCH |
| defer_sink 机制 | worker.py:930-950 + streaming_audio.py:935-941 | step 级批塔，bitwise 一致 |
| 塔图桶池 | tower_cuda_graph.py:44-61,159-213 | 7 桶 + 共享 pool + CPU max_seqlen |
| 塔图等价重写 | tower_cuda_graph.py:218-280 | bitwise 验证声明 |
| engine clock | openai/runtime_bridge.py:199-227 | tick 网格；yaml 未配=关 |
| 2-head 路由 | worker.py:1396-1480 | state argmax + route_position + mask 强制 |
| prev_state 键 | worker.py:1428-1440 | (session_id, incarnation) |
| EMPTY_ACTION_BLOCK | core/sched/omni_ar_scheduler.py:410-418 | action 段强制闭合 |
| base duplex_session 参数 | deploy/qwen3_omni_duplex.yaml | ttl/reaper/max_sessions |
| 三件套实测 | git e6a3ff1 提交说明 | 42→108 题/min、VCB 3h11m |
| LM CG 实测 | yaml:18 | 121→205 题/min |
| 并发/利用率 | git eb37611 提交说明 | 13.12%、3.6%、mtg_top50 坍缩 |
| stage0 扩容史 | git c7ce84a/cad0963 提交说明 | 2 副本 <20% → 7×TP1 |
| max_model_len 事故 | git c7ce84a 提交说明 | 65545 token broadcast 崩溃→131072 |

> 性能数字全部出自 git 提交说明与 yaml 注释（l30082962/7bcccbc/
> e6a3ff1/eb37611/c7ce84a 的实测记录），未独立复测；"剩余优化空间"
> 一节是本文的推断，已标明依据。

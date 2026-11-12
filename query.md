这个问题要查代码确认。先看音频增量机制和 forward 触发点：

growing_mel 只是音频特征侧的增量（demo 路径）。看核心的 forward 触发和 KV cache 逻辑：

音频侧确认了：每 240ms chunk ≈ 3 个音频 token，audio tower 有跨 session 批处理。现在确认 LLM 主干的 step 节奏和 KV 策略：

调度完全继承 vLLM V1（`super().schedule()`——KV cache 就是原生 PagedAttention）。再确认打断时的 KV 处理和 decode 节奏，闭环这个问题：

worker.py:392 有 KV 回滚重算的关键注释，读完整段：

"generation runs once per append" + context-roll 重算机制都找到了。查 roll 触发条件补全答案：

全部查清了。回答你的两个问题：

## 大 forward 频次：每 240ms chunk 一次生成

协议层强制（`runtime.py:189`）：*"each duplex append must contain exactly one 240 ms chunk; **generation runs once per append**"*——**音频每来 240ms，触发一次生成**。

这一次“生成”内部的结构（以一个 chunk 为单位）：

| 组成 | forward 次数 | 说明 |
|---|---|---|
| 音频增量 prefill | 1 次 | 新 chunk 的 ~3 个音频 token（13Hz mel → 240ms ≈ 3 帧）+ replay/封口结构 token，作为增量 append |
| agent token decode | 每 token 1 次 | legacy 每 chunk 恰 1 个 agent token；state_then_text 最多 3 个（span 内自回归，每 token 一次 decode forward） |
| audio tower | 1 次小 forward（3 帧） | 与 LLM 主干分开；有跨 session 批处理（`VLLM_OMNI_DUPLEX_TOWER_BATCH`，把所有 session 的 tower 工作攒成一个 batch，省掉逐 session 串行的 30-35ms） |

所以纯听（listen）的 chunk = 1 次小 prefill + 状态判断；说话的 chunk = 1 次小 prefill + 1~3 次 decode。引擎层面所有 session 的 prefill/decode 混在一个 batch step 里（vLLM V1 混合批，`super().schedule()` 原生调度）。

## KV cache：正常全复用，定期有一笔“滚动重算”

**正常路径（99% 的 chunk）**：完整走 vLLM 原生 PagedAttention——每 chunk 只是增量 append（`token_offset > 0` = "normal append to an already-prefilled prompt"，`worker.py:405`），`num_computed_tokens` 累计，历史 KV 块一个都不重算。

**例外：context roll（整窗重算）**——这是唯一丢 KV 的场景：

- 触发条件（`omni_ar_scheduler.py:1115-1140` `_maybe_roll_duplex_context`）：会话 chunk 数超过 `context_chunk_length + roll_batch + 1`
- 默认窗口 **300 chunks**（= 300 × 240ms ≈ **72 秒**音频上下文）+ 攒批 20（`serving_adapter.py:458`）——即**约每 321 个 chunk（77 秒）才触发一次**
- 触发时：prompt 裁剪成 `chunk0（system prompt + 首 chunk）+ 最近 300 chunks`，**`num_computed_tokens=0`，旧 KV blocks 释放，整个保留窗口重 prefill**（约 900-1500 token 量级的一次大 forward）
- 目的：防 `prompt_token_ids` 无限增长撑爆 `max_model_len`、限制注意力开销

**摊销账**：72 秒音频才付一次 ~1000 token 的重 prefill，平摊到每个 chunk 是 ~3 个 token 的额外 prefill 开销，相对每 chunk 本来就要做的增量 append（~3 音频 token + 结构 token），翻倍但不到一个量级。

<options>
    <option>看 CHUNK_LAYOUT.md 的序列结构文档（roll 机制有专门章节）</option>
    <option>算一算实际评测时的 forward/token 吞吐对账</option>
    <option>清楚了</option>
</options>

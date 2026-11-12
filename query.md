合理要求——上一轮只讲了骨架。现在把主循环和每个缓存/预算机制逐行过一遍：

逐项挖完了，包括参数默认值和主循环全貌。这次是**每个加速点的完整细节**：

---

# fast 推理的全部加速细节（`ablation/speak-3-text-tokens` 分支）

## 0. 全景：主循环的“双相位”节奏

```python
# fast.py:1444-1476 —— Phase 1: 消耗源音频
while not audio_exhausted and not _hit_stop():
    if prev_state in CONTROL_STATES:
        assumed = "idle" if prev_state == "idle" else "listen"
        turn_state = prefill_control_run(assumed)   # ← 投机批量
        if turn_state in ("speak", "interrupt", "text"):
            while ... prev_state in ("speak", "interrupt", "text") ...:
                ar_one_chunk()                       # ← 逐 chunk AR
# fast.py:1480 —— Phase 2: 源音频耗尽后零波形续 decode（always-on）
while _n_pad() < max_chunks and not _hit_stop(): ...
```

加速只发生在**控制态**（listen/idle 段）；speak/text 段两套实现完全一样（逐 chunk AR）。所以**加速比 = listen/idle 占比的函数**：全 listen 的 WAV 收益最大，全程说话的 WAV 加速≈1。

## 1. 投机控制段 prefill——核心加速的全部细节

### 1a. draft 什么、怎么 draft

```python
# fast.py:209-212
# control token can be DRAFTED and prefilled. speak/text cannot be drafted
# (text depends on the previous content token; speak is a turn point we're
# trying to FIND, not assume). None (start) and text_break draft listen.
CONTROL_STATES = (None, "listen", "idle", "text_break")

# fast.py:932-936 —— draft 的槽位 token 由当前状态决定
def _agent_slot_token_for_state(state):
    if state == "idle":
        return idle_id
    return listen_id  # None, listen, text_break -> draft listen
```

draft 塞进每个 chunk 的 agent 槽（`build_full_chunk_tokens(..., agent_token_id=drafted_token)`），chunk 布局不变——draft 的本质是**假设槽位 token，提前把它写进输入序列**。

### 1b. K 的四层预算（不是拍脑袋的常数）

```python
# fast.py:945-975 —— 语义 cap → 音频余量 cap → pad 最小值 cap → 总 token 预算 cap
if audio_remaining > 0:
    k_limit = min(max_control_prefill, audio_remaining)      # ① 不跨源/pad边界
else:
    pad_allowance = max(0, PAD_MIN_CHUNKS - n_pad_committed) # ② pad 阶段的回合约束
    k_limit = min(max_control_prefill, pad_allowance,
                  max(0, max_chunks - n_pad_committed))
...
projected_total = len(all_token_ids) + drafted_token_count + len(toks)
if projected_total > max_prefill_tokens and chunk_tokens:    # ③ 16384 总预算
    break
```

- `max_control_prefill=64`（CLI 默认，fast.py:2014）
- `max_prefill_tokens=16384`（fast.py:2016）——“long prefixes automatically shrink K”：前缀越长 K 自动越小
- **K=1 正确性兜底**（fast.py:980 注释）：*"If even one chunk no longer fits, allow K=1 as the AR-equivalent correctness fallback; the token budget limits speculation, never forward progress"*——预算只限投机深度，**任何情况下都不阻断前进**。

### 1c. 一次 forward、一次 state_head、K 个位置

```python
# fast.py:1000-1006
full_seq_ids = all_token_ids + drafted_ids            # 提交前缀 + K 个 draft chunk
logits, hidden = ref.forward_step(model, input_ids, mod_ids, audio_feats, ...)
# state_head over the full sequence ONCE (2-head); index per position below.
if enable_state_head:
    state_logits = model.thinker.state_head(hidden)   # 整序列算一次，K 个位置按 index 读

# fast.py:1008-1013 —— 绝对槽位 = 前缀长 + 前面 draft chunk 累计 + agent_pos_in
positions.append(cursor + chunk_agent_pos_in[i])
```

K 个 chunk 的全部决策读数（LM logits + state_head logits）**来自这一次 forward**。`agent_pos_in_chunk` 指向 `<|agent_text_start|>`——它的 hidden 对齐训练 shift，预测的就是 agent 槽。

### 1d. 提交/丢弃规则（因果性是唯一的正确性依据）

```python
# fast.py:1072-1081
if new_state == assumed_state:
    _commit_chunk(chunk_tokens[i], ...)   # draft 确认：token 原样提交（槽位本来就是 drafted_token）
    committed_audio += 1
    continue
# First mismatch = turn point. Commit chunk i with its REAL routed token ...
# Chunks i+1..k-1 are DISCARDED (their context includes the wrong drafted
# token at i, not emitted_token).
```

- 转折点位置的预测**有效**：位置 i 的 hidden 只依赖前缀（is_causal）+ 本 chunk 自己的前段，**drafted token 是被预测对象不是输入**——所以 mismatch 位置的 routed 结果直接可用，只需用 `emitted_token` 重建该 chunk 的 token 序列再提交。
- 音频游标只前进 `committed_audio`（fast.py:1223）——**被丢弃 chunk 的音频留给下一轮**，音频消费与参考逐 chunk 对齐。
- 全 K 接受 → `return None` → 主循环再来一轮 draft。长 listen 段 = 连续多轮 draft，每轮吃掉最多 64 个 chunk。

## 2. 音频侧：预计算 + 虚拟组序列 + pad 缓存

### 2a. 全量预计算（bit-identical 论证）

```python
# fast.py:384-399
def precompute_all_audio(mel_enc, aud_enc, wav):
    """...the reference's Phase-1 loop (80ms wav chunks -> mel_enc.push_wav ->
    buffer 3-row groups) with the run_one_chunk calls removed...
    Bit-identical features to the reference: StreamingAudioEncoder.push is causal
    (refuses to fabricate lookahead), and the LM forwards the reference interleaves
    do NOT affect the audio encoder's state."""
```

与参考脚本**同一个 push 序列**（80ms → mel → 攒 `AUDIO_TOKENS_PER_CHUNK=3` 行成组），只是把“每 chunk 夹一次 LM forward”拆掉、音频一次性算完。等价性靠两个事实：encoder 因果 + LM forward 不污染 encoder 状态。

### 2b. 虚拟组序列：源组 → 懒生成 pad 组

```python
# fast.py:705-724 —— zeros 过同一个 streaming frontend，不是伪造向量
def _ensure_pad_groups(n_needed):
    while len(pad_row_groups) < n_needed:
        feats = mel_enc.push_wav(zero_chunk, aud_enc)   # 80ms 数字零
        ...攒满 3 行入 pad_row_groups...

def _group_at(gi):   # 源组在前，pad 组在后；永远有音频（always-on decoding）
    if gi < len(audio_row_groups):
        return audio_row_groups[gi], "audio"
    _ensure_pad_groups(gi - len(audio_row_groups) + 1)
    return pad_row_groups[gi - len(audio_row_groups)], "silence_pad"
```

### 2c. pad 的确定性 → 跨 draft 复用

```python
# fast.py:697-701
# Zeros never raise the running mel max, so pad rows are causal + deterministic
# and CACHED across discarded prefill drafts (a draft's pad pushes see the
# identical encoder prefix).
```

zeros 不会改变 mel 的 running max → pad 行只依赖 encoder 前缀 → **draft 被丢弃时，它里面已经生成过的 pad 组直接命中缓存**，音频计算零浪费。

## 3. bookkeeping 合一：一条提交路径

`_commit_chunk`（fast.py:731）被**两条路径共用**（draft 确认/转折点 + AR 逐 chunk）——timeline、prob_log、文本缓冲、acceptance latch、routing invariants 全在一处，防止 fast/AR 两套记账漂移。音频对齐还有专门的不变量守卫（fast.py:846-856）：`audio_group_idx` 必须等于“前面消费的组数”，专抓“说话期间/打断时吃错音频组”这类 bug。

## 4. 本分支（3T）的反向细节：槽内循环是**负优化区**

SPEAK 转折点读变长文本槽时（fast.py:1110-1140）：

```python
for _j in range(1, speak_text_n):
    _loop_ids = stt_seq_ids + speak_text_ids          # 全前缀 + state + 已读 text
    logits, hidden = ref.forward_step(model, _loop_input, ...)   # 每个文本 token 一次 full-prefix forward
```

**每个 text token 都是一次全量 forward（无 KV cache）**——3T 相对 N=1 最多 3 倍 forward 开销，这是 ablation 为语义付出的代价（换取一个 SPEAK chunk 决策携带最多 3 个 text token）。加速设计在转折点内部**主动让位**——因为 text token 之间强依赖，无法 draft。

## 5. 明确不做的优化（同样重要）

- **不用 KV cache**（`use_cache=False`）：两个原因——① `--verify_ar` 的 EXACT 合约要求与参考逐位一致，KV cache 引入的数值差会破坏它；② draft 丢弃语义与 KV cache 的交互（丢弃后回滚缓存）会让实现复杂度爆炸，而“合并 forward”已经拿到了大头收益。
- **audio tower 批处理留给 WS4**（fast.py:399 注释：*"(WS4 scope: only the tower-forward grouping moves, inside StreamingAudioEncoder.push)"*）——本脚本不动 encoder 内部。

## 6. 加速比的定量直觉

第 i 个 chunk 的 full-prefix forward 是 O(i×~10) token（每 chunk ≈ 3 音频 + 7 结构 token）。参考脚本总计算量 Σ O(i) ≈ **O(L²/2)**；fast 按平均 K̄ 个 chunk 合并，总计算量 ≈ **O(L²/(2K̄))** + 少 K̄-1 倍的 kernel launch 固定开销。**K̄ 是 listen 段实际 draft 命中深度**——实测加速比就取决于它（`--verify_ar` 的 TIMING 输出里有 generation-only 墙钟对比）。

<options>
    <option>实测跑一次 --verify_ar 看加速比和等价性报告</option>
    <option>清楚了</option>
</options>

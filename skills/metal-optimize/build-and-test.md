# 构建与测试（Metal LLM）

> **配套 SKILL.md 的 sub-doc**：build 命令、模型导出、性能测试脚本。做完 `kernel-dev-and-optimize` / `graph-fusion` / `runtime-scheduling` 里描述的改动后，回到这里跑测试。

---

## ⚠️ 每次 Metal 改动之后的强制验证流程（最重要）

**任何 Metal 改动都必须走完这套流程再评估性能，否则大概率是假信号。** 这条不是建议，是硬性规则。

### Step 0. 强制重编译，别相信 make 增量

Shader 是嵌入 `.hpp` 里的 C++ 字符串，改完 `.hpp` 后 `make` 有时**只重编产物但不重新 link `libMNN.dylib`**（时间戳判断问题），运行时仍加载旧 shader。

```bash
# 改完 shader 或 .mm 后，touch 强制标记 dirty，或 -B 全量重编
touch source/backend/metal/MetalAttentionShader.hpp     # 或改动的 shader/mm
cd build && make -j8 llm_demo
# 或强力版：
cd build && make -j8 -B llm_demo
```

如果测试结果诡异（比如"改了 kernel body 但行为没变"），第一步先 `ls -l build/libMNN.dylib` 看链接时间是否最新，不是就 `make -B`。

**新增源文件后必须重跑 `cmake ..`。** build 目录的源文件列表是 configure 时 GLOB 出来的，新增 `.mm/.hpp` 不会被自动纳入 —— 表现为链接期 `Undefined symbols`（2026-07-30 前科：`MetalReplayProxy` / `metalReplayEmit` / `metalReplayValidate` 全缺）。更坏的情况是 link 失败后 `libMNN.dylib` 已被删掉，而你以为"构建成功过"。

```bash
cd build && cmake .. && make -j10 llm_demo llm_bench
```

### Step 0.5 测量前的二进制新鲜度断言（血泪，必做）

**过期 build 不会报错，只会静默给出错误的性能数字。** 2026-07-30 前科：整轮 MNN-vs-MLX 对比（4 模型 × 3 prompt 长度）跑在一个隔了一天的 `libMNN.dylib` 上，0.6B decode 偏低 6~8%，还据此立项排查了一个不存在的"-9% 回归"（建 worktree、编译 baseline、正反序配对、二分两轮，全部白做）。

```bash
# 1) 时间戳必须晚于你最后一次改动
ls -l build/libMNN.dylib

# 2) 关键符号断言：改动引入的新 kernel/函数名必须出现在产物里
strings build/libMNN.dylib | grep -c prefill_flash_attn_nax   # 应为非 0
```

判据：**只要"当前 HEAD 引入的某个新符号"在 `libMNN.dylib` 里找不到，这一轮所有数字作废。** 反向也成立——排查"疑似回归"时，先做这个断言再去建 worktree 二分。

⚠️ 顺带一个易踩的输出格式口径：HEAD 的 `llm_bench -pg <pp>,<tg>` **分列报告 prefill / decode 两个速度**（`-kv` 已废弃，`-p A -n B -kv true` == `-pg A,B`）。历史 WIP 二进制的 `-pg` 输出格式与 HEAD 不同——发现输出列数/口径和预期不符，本身就是"跑的不是当前产物"的信号，回到本节做新鲜度断言。

### Step 1. 清 Metal pipeline binary cache

Metal 会把 pipeline JIT 结果缓存到 `tmp/mnn_cachefile.bin`（launch 目录相对路径）。改 shader 后 pipeline key 可能没变（宏组合相同），Metal 会加载旧 binary → 观察到"改了 shader 但完全没生效"。

```bash
find . -name "mnn_cachefile.bin" -delete
# 常见位置: build/tmp/mnn_cachefile.bin, ./tmp/mnn_cachefile.bin (llm_demo launch dir)
```

### Step 1.5 新导出模型：先造 greedy config，禁止用导出默认 config 对拍

`llmexport.py` 产出的默认 `config.json` 是 **`backend_type: cpu` + `sampler_type: mixed` + `temperature 0.8`**——拿它跑 `llm_demo` 做"对拍/自拍"，得到的是**CPU 上的随机采样**：自拍必然 DIFFERS、从第 1 个 token 就分叉，且任何 Metal env 开关都"看似无效"。2026-07-29 的前科：splitkv 越界修复后的验证被它污染，误报出一个"p4096 prefill 非确定性"的假 bug，差点为此新开排查。

```bash
# 新模型落地第一步（llm_bench 因有 -a metal 不受影响，llm_demo 全部用这份）
python3 - <<'EOF'
import json, sys
p = '/path/to/model/config.json'
d = json.load(open(p))
d['backend_type'] = 'metal'; d['sampler_type'] = 'greedy'; d['temperature'] = 0.0
json.dump(d, open(p.replace('config.json', 'config_mtl_greedy.json'), 'w'), indent=4)
EOF
```

**判据**：自拍（同 config 连跑两次）DIFFERS 时，第一反应先 `grep sampler_type <config>`，再怀疑代码。

### Step 2. 正确性验证矩阵（必须全部跑）

**只测速度不测正确性 = 假信号。**

强制使用 `sampler_type: greedy, temperature: 0.0, top_k: 1` 的 config（跨 run byte-identical 是黄金标准）。

对每一次改动，跑满这套矩阵：

| 维度 | 覆盖点 | 为什么 |
|---|---|---|
| **Prompt 长度** | 短 (~50 tok) + 中 (~512 tok) + 长 (~2048 tok) | 触发不同 kernel 路径（mShortSeq / mQkSimdMatrix / mQkTensorMatrix / mFlashAttnPrefill） |
| **FA on / off** | `MNN_ENABLE_FLASH_ATTN_PREFILL=1` 和 `=0` | 决定走 flash-attn 还是三段 pipeline (prefill_qk[_tensor] + softmax + prefill_qkv[_tensor])。两条路径都要正确 |
| **CAUSAL_TRI（数据驱动，2026-07-31 起无 env）** | 用真实 mask 张量 vs 标量哨兵 mask 两类模型覆盖 | causal-tri/bound 现由 `mCausalLayout`（inputs[3] 形状）自动 gate，非手动 env。**任何 attention/softmax 改动都要同时覆盖 causal（标量 mask）与非 causal（真实张量 mask）两条路径**，见下方 § "Attention causal 假设" |
| **每一个新增 env var** | 默认（不设）+ 每个显式值都跑一遍 | Env 只在 static 初始化时读一次（`static const int kX = getenv(...)`），不同值 = 完全不同分支 |
| **至少 2 个模型 shape** | `head_dim ∈ {64, 128, 256}` × `group_size ∈ {1, 2, 4, 8}` | Qwen3-0.6B (D=128, G=2)、Qwen3-4B (D=128, G=4)、Qwen3.5-2B (D=256, G=4) — 每个都可能踩不同 layout / stride 分支 |
| **Mask 语义（数据驱动）** | 非 causal 模型（SWA / prefix LM / bidirectional）**无需再设 env** | Metal 2026-07-31 起从 mask 张量形状自动判定：真实张量 ⇒ 逐元素 honor、关全部 causal 优化；标量/无 mask+kvcache ⇒ causal。若非 causal 仍乱码，查 `gen_attention_mask` 是否为该模型产出真实张量 mask（非误走标量）。详见下方 § "Attention causal 假设"。 |

**判据**：跟 baseline 前 N (≥ 20) tokens byte-identical，或至少输出语义合理（无乱码 / 无异常重复 / 无语言跳变）。

Baseline 选取原则：
- **首选** CPU 后端 greedy 输出（layout 无关，最干净的 oracle）
- **次选** 已知正确的 Metal path（比如改 `prefill_qkv_tensor` 时用 FA on 的输出对拍）

### Step 3. 全模型正确性 sweep（模板）

```bash
MAX_TOKENS=30
for M in qwen3-0.6b-head-b32 qwen3-4b-head-b32 qwen3.5-0.8b-head-b32 qwen3.5-2b-head-b32; do
  CFG=/Users/jiuqi/models/${M}/config_mtl_greedy.json
  for FA in 1 0; do
    echo "=== ${M} FA=${FA} ==="
    MNN_ENABLE_FLASH_ATTN_PREFILL=$FA \
      DYLD_LIBRARY_PATH=build:build/express build/llm_demo \
      "$CFG" /tmp/prompt_2048_oneline.txt $MAX_TOKENS 2>&1 \
      | awk '/^prompt file is/{f=1;next} /^#####/{f=0} f' | head -3
    echo
  done
done
```

新增 env var 时把外层循环再加一维（`for E in 0 1 default; do ...`）。

### Step 4. 只在 Step 2/3 全过后才跑性能对比

先看正确性，正确性 OK 后才有理由测 t/s 数字。跑 3-rep A/B (WARMUP + SHUFFLE) 消噪声，见下面 § 性能测试。

---

## 编译

```bash
# 标准 Metal + LLM 编译
mkdir -p build && cd build
cmake .. -DMNN_METAL=ON -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
make -j8 llm_demo llm_bench MNN

# 带 profiling 编译（Step 1）
cmake .. -DMNN_METAL=ON -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_METAL_OP_PROFILE=ON
make -j8 llm_demo

# 带 converter（导出模型需要）
cmake .. -DMNN_METAL=ON -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_CONVERTER=ON
make -j8 llm_demo MNNConvert
```

## 模型导出

```bash
cd transformers/llm/export
python llmexport.py --export mnn \
    --path /path/to/HuggingFace/model \
    --mnnconvert /path/to/build/MNNConvert
```

## 性能测试

**用 `llm_bench` 而不是 `llm_demo` 测性能，且必须带 `-pg`。** `-pg <pp>,<tg>`：prefill pp 个 token 后复用该 KV cache 继续 decode tg 个 token（同 llama-bench 的 -pg），**分列报告 prefill / decode 两个速度**。`-kv` 已废弃（`-p A -n B -kv true` == `-pg A,B`）；单独的 `-p` / `-n` 是 prefill-only / decode-only 口径，默认值 512/128 会额外生成独立测试，**只想跑 -pg 时须加 `-p 0 -n 0` 压掉**。`-pg` 可重复传，多组累加。

```bash
cd build

# Metal 后端（-a metal 直接指定，无需改 config.json）
./llm_bench -m /path/to/model/config.json -a metal -p 0 -n 0 -pg 512,128 -rep 3

# CPU 后端对比
./llm_bench -m /path/to/model/config.json -a cpu -t 4 -p 0 -n 0 -pg 512,128 -rep 3

# 不同 prompt 长度（一次跑多组）
./llm_bench -m /path/to/model/config.json -a metal -p 0 -n 0 \
    -pg 64,64 -pg 512,128 -pg 2048,128 -rep 3

# FA A/B
./llm_bench -m /path/to/model/config.json -a metal -p 0 -n 0 -pg 512,128 -rep 3 -fa 0
./llm_bench -m /path/to/model/config.json -a metal -p 0 -n 0 -pg 512,128 -rep 3 -fa 1

# 长 prompt + 内存受限：chunk + FA + KV int8
# config.json 加 "chunk": 512, "attention_mode": 10
```

## 正确性验证（LLM 场景）

```bash
# CPU 和 Metal 同 prompt + temperature=0，前 N token 应一致
# config 中设 "temperature": 0.0

# CPU 基线
./llm_demo config_cpu.json prompt.txt 30

# Metal 对比
./llm_demo config_metal.json prompt.txt 30

# FA A/B 对比（同一 config）
MNN_ENABLE_FLASH_ATTN_PREFILL=0 ./llm_demo config_metal.json prompt.txt 30 > off.log
MNN_ENABLE_FLASH_ATTN_PREFILL=1 ./llm_demo config_metal.json prompt.txt 30 > on.log
diff off.log on.log
```

## Attention causal 假设（⚠️ 加载非标准模型前必读）

Metal 后端**两条** prefill attention 路径**都硬编码了 "attention mask 是 causal lower-triangular" 的假设**，运行时不做验证：

- **三段路径**（`prefill_qk[_tensor]` + `softmax` + `prefill_qkv[_tensor]`）：CAUSAL_TRI host 只 dispatch 对角线以下 tile；CAUSAL_BOUND softmax 只归约 valid prefix + zero-pad；AV 用 `av_k_upper` 早退。**违反假设 → 上三角"应有效"位置被静默丢弃**。见 `MetalAttentionShader.hpp:558/651` 的 `Assumption:` 注释。
- **FA 路径**（`prefill_flash_attn`）：`in_bounds = (kv_col_abs <= q_abs + kv_valid_offset)` hard-code causal。见 `MetalAttention.mm:531`。

**因此以下模型加载 Metal 后端会静默错**（不崩、不 warning、只是输出乱）：

| 模型类别 | 举例 | 症状 |
|---|---|---|
| Sliding Window Attention | Mistral 7B v0.1, Gemma-2, Ministral | 短 prompt 可能对，超过 window size 后开始漂移 |
| Mixed window（层交替） | Gemma-2（每层交替 SWA / full）| 层内 window 边界后开始错 |
| Prefix LM | Baichuan-Base 前缀部分、UL2 | 从第一 token 就错 |
| Encoder-decoder cross-attention | T5、UL2、Whisper | 完全不适用 |
| BERT-family bidirectional | 任何 encoder 模型 | 完全不适用 |

**准入检查（导入新模型前跑一次）**：

```bash
# 数据驱动检测（2026-07-31 起）：causal 由 gen_attention_mask 产出的 mask 形状决定
# —— 标准 causal 模型发标量哨兵 mask，非 causal 模型发真实张量 mask，Metal 自动分流。
# 无需 A/B env 对拍。直接与 CPU 后端 greedy 对拍前 20 token：
DYLD_LIBRARY_PATH=build:build/express build/llm_demo <cfg_metal> <prompt> 20 > /tmp/a.log 2>&1
DYLD_LIBRARY_PATH=build:build/express build/llm_demo <cfg_cpu>   <prompt> 20 > /tmp/b.log 2>&1
diff <(awk '/^prompt file is/{f=1;next}/^#####/{f=0}f' /tmp/a.log) \
     <(awk '/^prompt file is/{f=1;next}/^#####/{f=0}f' /tmp/b.log)
# ✓ 无 diff → Metal 输出与 CPU 一致，正确
# ✗ 有 diff → 查 gen_attention_mask 是否为该模型走了正确分支（真实张量 vs 标量），
#             根因多在 mask 生成/导出侧
```

**推荐操作**：
- 加载 **Qwen / Llama / Phi / DeepSeek / Yi / Baichuan-Chat** 等纯 causal LLM：直接跑，标量哨兵 mask + causal 优化全开
- 加载 SWA / prefix / bidirectional 模型：**无需再设任何 env**（2026-07-31 起数据驱动）；真实张量 mask 自动触发逐元素 honor、关掉 causal-tri/bound/FA。前提是 `gen_attention_mask` 为该模型产出真实张量 mask（SWA 走 `attention_type=mix` 双平面；确认导出 config 正确）
- 不确定模型是不是 causal：读 HF `config.json` 里有没有 `sliding_window` / `attention_bias` / `is_encoder_decoder` 字段

**注意**：causal-tri/bound 及 FA-v1/faNax 的 causal 假设现由 `mCausalLayout`（inputs[3] 形状）统一 gate。真实张量 mask 会一并关掉 FA（含 FA-v1 那条以前无 opt-out 的路径），不再需要手动关 `MNN_ENABLE_FLASH_ATTN_PREFILL`。


# 构建、测试、性能基线（Metal LLM）

> **配套 SKILL.md 的 sub-doc**：build 命令、模型导出、性能测试脚本、基线数据、文件索引。做完 `kernel-basics` / `llm-optimizations` 里描述的改动后，回到这里跑测试。

---

## ⚠️ 每次 Metal 改动之后的强制验证流程（最重要）

**任何 Metal 改动都必须走完这套流程再评估性能，否则大概率是假信号。** 这条不是建议，是硬性规则 —— 违反的代价见 § 真实教训。

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

### Step 1. 清 Metal pipeline binary cache

Metal 会把 pipeline JIT 结果缓存到 `tmp/mnn_cachefile.bin`（launch 目录相对路径）。改 shader 后 pipeline key 可能没变（宏组合相同），Metal 会加载旧 binary → 观察到"改了 shader 但完全没生效"。

```bash
find . -name "mnn_cachefile.bin" -delete
# 常见位置: build/tmp/mnn_cachefile.bin, ./tmp/mnn_cachefile.bin (llm_demo launch dir)
```

### Step 2. 正确性验证矩阵（必须全部跑）

**只测速度不测正确性 = 假信号。** 见 § 真实教训里的 P0 flash-attn small-gate 案例。

强制使用 `sampler_type: greedy, temperature: 0.0, top_k: 1` 的 config（跨 run byte-identical 是黄金标准）。

对每一次改动，跑满这套矩阵：

| 维度 | 覆盖点 | 为什么 |
|---|---|---|
| **Prompt 长度** | 短 (~50 tok) + 中 (~512 tok) + 长 (~2048 tok) | 触发不同 kernel 路径（mShortSeq / mQkSimdMatrix / mQkTensorMatrix / mFlashAttnPrefill） |
| **FA on / off** | `MNN_ENABLE_FLASH_ATTN_PREFILL=1` 和 `=0` | 决定走 flash-attn 还是三段 pipeline (prefill_qk[_tensor] + softmax + prefill_qkv[_tensor])。两条路径都要正确 |
| **每一个新增 env var** | 默认（不设）+ 每个显式值都跑一遍 | Env 只在 static 初始化时读一次（`static const int kX = getenv(...)`），不同值 = 完全不同分支 |
| **至少 2 个模型 shape** | `head_dim ∈ {64, 128, 256}` × `group_size ∈ {1, 2, 4, 8}` | Qwen3-0.6B (D=128, G=2)、Qwen3-4B (D=128, G=4)、Qwen3.5-2B (D=256, G=4) — 每个都可能踩不同 layout / stride 分支 |

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
make -j8 llm_demo MNN

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
    --mnnconvert /path/to/build/MNNConvert \
    --quant_block 32
```

## 性能测试

```bash
cd build

# Metal 后端：修改 config.json 中 backend_type 为 "metal"
./llm_demo /path/to/model/config.json /path/to/prompt.txt 30

# CPU 后端对比
./llm_demo /path/to/model/config_cpu.json /path/to/prompt.txt 30

# 不同 prompt 长度测试
./llm_demo /path/to/config.json /path/to/short_prompt.txt 50   # 短 prefill
./llm_demo /path/to/config.json /path/to/512.txt 30            # 中等 prefill
./llm_demo /path/to/config.json /path/to/1024.txt 30           # 长 prefill

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

## 真实教训（为什么强制验证流程存在）

**案例 1：P0 flash-attn small-model gate（2026-07 M5）**
基于 M5 pp2048 A/B "no_flash_attn +7.1% prefill" 数据，加了小模型自动关 FA 的 gate。合入前一个 correctness 抽查发现 Qwen3-0.6B 输出乱码 `个 https:// 中文文告 thái`。**"+7.1% 更快"其实是 FA off 路径本身有 bug，写错 layout 直接吐乱码，"快"只是因为少写了正确数据**。

Root cause: `prefill_qkv_tensor` kernel 缺 `#ifdef ATTENTION_C4` 分支（长期潜伏），只有 c4-head 导出模型 + M5+ tensor API 路径 + FA off 三个条件同时命中才触发。前 3 次尝试 fix 反复验证"没生效"，原因是 make 增量没 relink libMNN.dylib（Step 0）+ Metal pipeline cache 命中旧 binary（Step 1）。

**教训**：
- **没有正确性验证的"性能提升"不能信** —— 假信号率极高。今天 P0-D 5 次尝试 4 次是假信号。
- **改 shader 一定要 `touch header + make -B` + 删 pipeline cache** 后测。
- **必须覆盖 FA on/off × 多模型** —— 单一路径过了不代表另一路径没坏。

**案例 2：M3 Pro fusion 诊断（2026-07）**
第一版脚本每场景 3-rep 全跑完再切下一场景，第一个跑的 baseline 吃冷启动税 → 后面场景全比 baseline 快 5-7%，包括理论上不该有效应的 `no_fused_Q4_gemm`（M3 Pro 硬 gated off）。用户 catch 到这个矛盾信号，`v2` 改成 outer=rep / inner=case + WARMUP + SHUFFLE 后，delta 大幅缩小。**性能测量的 process-level cold start 效应比 kernel 差异大**。

---

## 性能基线（Qwen3-0.6B Q4, Mac M4）

| 指标 | CPU (4 线程) | Metal | 加速比 |
|------|------------|-------|--------|
| 短 prompt (15 tok) prefill | 424 tok/s | 920 tok/s | **2.2x** |
| 短 prompt decode | 188 tok/s | 193 tok/s | 1.03x |
| 长 prompt (514 tok) prefill | 1036 tok/s | 4033 tok/s | **3.9x** |
| 长 prompt decode | 142 tok/s | 260 tok/s | **1.8x** |
| 长 prompt (999 tok) prefill | 1089 tok/s | 2601 tok/s | **2.4x** |
| 长 prompt decode | 101 tok/s | 141 tok/s | **1.4x** |

---

## 文件索引

| 文件 | 职责 |
|------|------|
| `source/backend/metal/ConvSimdGroupShader.hpp` | GEMV/GEMM shader 字符串（deferred dequant, 双 simdgroup, pre-scaling, GATE_UP_FUSED, QKV_FUSED） |
| `source/backend/metal/MetalConvolution1x1.mm` | Conv1x1 dispatcher，buffer 分配，kernel 选择，Gate/Up 和 QKV fusion setup/encode |
| `source/backend/metal/MetalConvolution1x1.hpp` | Conv1x1 成员变量（deferred dequant buffers, Gate/Up fusion, QKV fusion） |
| `source/backend/metal/MetalAttention.mm` | Attention dispatcher，GQA group_size 选择，flash-attn prefill eligibility/dispatch |
| `source/backend/metal/MetalAttention.hpp` | Attention 成员（`mFlashAttnPrefill`, `mKernel_flashAttn`）|
| `source/backend/metal/MetalAttentionShader.hpp` | Fused attention shader（decode_qk_softmax） |
| `source/backend/metal/MetalFlashAttnShader.hpp` | Fused prefill flash-attention shader（`gPrefillFlashAttn`）|
| `source/backend/metal/MetalLayerNorm.mm` | RMSNorm kernel 选择 |
| `source/backend/metal/MetalRope.mm` | RoPE Metal 实现（inv_freq fusion） |
| `source/backend/metal/MetalBinary.mm` | Binary op 实现，Gate/Up fusion 关系发现（从 MUL_SILU 输入） |
| `source/backend/metal/MetalBackend.mm/hpp` | Op profiling、Gate/Up 注册匹配、QKV 分组注册匹配（`registerConv1x1ForQKV`、`matchQKVFusions`） |
| `source/backend/metal/MetalDefine.h` | MNN_METAL_OP_PROFILE 宏 |
| `source/backend/cpu/CPURoPE.cpp` | RoPE CPU 实现对齐 |
| `schema/default/MNN.fbs` | RoPE op schema（hasInvFreq） |
| `tools/converter/source/optimizer/postconvert/RemoveDeadShapeOp.cpp` | 死代码消除 pass |
| `tools/converter/source/optimizer/postconvert/FuseTransformerC4.cpp` | QKV projection 图重排 |
| `transformers/llm/export/utils/transformers.py` | LLM 导出（RoPE unsqueeze 修复） |
| `transformers/llm/export/utils/custom_op.py` | FusedRoPE 导出 |
| `transformers/llm/engine/src/llm.cpp` | attention_mode 默认值、chunk 处理 |
| `docs/transformers/llm.md` | Metal attention_mode 配置说明 |

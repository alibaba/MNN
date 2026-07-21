# 构建、测试、性能基线（Metal LLM）

> **配套 SKILL.md 的 sub-doc**：build 命令、模型导出、性能测试脚本、基线数据、文件索引。做完 `kernel-basics` / `llm-optimizations` 里描述的改动后，回到这里跑测试。

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

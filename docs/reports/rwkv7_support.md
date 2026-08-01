# RWKV7 模型支持工作报告

- **日期**：2026-08-01
- **分支**：`feature/support_rwkv7_mdataplus_fwaxl3`
- **验证模型**：`RWKV7-Goose-World2.8-0.1B-HF`（191M 参数，`model_type: rwkv7`）

## 1. 任务

为 MNN 添加 RWKV7 模型的导出（Python）与 C++ 推理支持，保证 C++ 推理结果与 HuggingFace 参考实现一致。按 `skills/support-new-llm` 的 TDD 流程执行（步骤 1→2→6→3→4）。

## 2. 模型架构分析

RWKV7 是线性注意力 RNN（Tier 6 全新架构），无 RoPE、无标准 QKV attention：

- **每个 block**：`[pre_norm(仅 layer0)] → attn_norm → attn → 残差 → ffn_norm → ffn → 残差`
- **Attention（RWKV7Attention）**：
  - token-shift 混合：`x_? = x + (x_{t-1} - x) · mix_?`（mix ∈ {x_r, x_w, x_k, x_v, x_a, x_g}，带状态）
  - 投影：`r_proj/k_proj/v_proj/o_proj` + 嵌套 LoRA（`w_lora/a_lora/g_lora/v_lora`，v_lora 仅 layer1+）
  - 每 head 递归（fla fused_recurrent 语义）：
    ```
    kk  = l2norm(k · k_k)
    k'  = k · (1 + (a - 1) · k_a)
    S   = exp(w) ⊙ S − (kk·a) ⊗ (kkᵀS) + k' ⊗ v      （S: [dk, dv] 状态）
    o   = g ⊙ (GroupNorm(Sᵀr, eps=head_dim·norm_eps) + v · Σ(r·k'·r_k))
    ```
  - 跨层 `v_first`：layer0 的 v 被后续所有层复用（`v = lerp(v, v_first, σ(v_lora(xv)))`）
- **FFN（RWKV7FeedForward）**：token-shift 混合（x_k）+ `value(sqrelu(key(mixed)))`
- **Tokenizer**：RWKV World 字节级贪心最长匹配 trie（65536 词表，special id 0 与 65530=`\n\n`）

## 3. 实现内容

### Python 导出侧

| 文件 | 内容 |
|---|---|
| `utils/model_mapper.py` | `regist_rwkv7`：config/model/decoder/linear_attention 映射 |
| `utils/transformers.py` | `RWKV7Attention`（mixing op + 递归 op 双路径：test 路径完整 torch 参考实现，ONNX 路径走自定义算子）、`RWKV7Mlp`、Decoder 支持 layer0 `pre_norm` |
| `utils/tokenizer.py` | RWKV World 词表以 TIKTOKEN 格式导出（C++ Tiktoken 即 trie 最长匹配，语义一致） |
| `llmexport.py` | `config.json` 的 `tokenizer_file` 跟随实际导出文件名 |

自定义算子设计（复用既有 `FusedLinearAttention` 框架，新增两个 attn_type）：

- `rwkv7_mixing`：qkv = N 份 hidden 拼接 `[B, N·H, L]`，conv_weight = `[mix, 1-mix]` 逐通道 2-tap 带状态卷积，实现 token-shift
- `rwkv7`：qkv 段 `[r, w, k, a, v, g]`，params 输入 `[gnorm_w, gnorm_b, r_k, k_k, k_a, eps]`，内部完成 kk 归一化、k′ 修正、递归、GroupNorm、gate 修正；递归状态恒为 fp32

### C++ 推理侧（`CPULinearAttention`）

- 新增 `rwkv7_mixing()` / `rwkv7()` 实现（多线程按通道/head 并行，支持 NC4HW4 输入输出）
- `onResize` 按 attn_type 分配状态：`rwkv7` 递归状态 `[B, H, dk, dv]` fp32
- 状态快照/回滚、prefix cache 等既有机制自动适用

## 4. 过程中发现并修复的 4 个通用导出管线 bug

均已沉淀到 `skills/support-new-llm/common-pitfalls.md` §15–18：

1. **§15 MNNConvert 二进制过期**：`build/MNNConvert` 早于 schema 更新时，JSON round-trip 会静默丢弃新字段（`IDSTQuan.scaleStorage`）→ fp16 alpha 被按 fp32 解读 → 所有量化 Linear 输出归零。只增量编译 `llm_demo` 不会更新 MNNConvert。
2. **§16 GPU 量化结果损坏**：torch 2.13 + CUDA 13 组合下 GPU 量化可能返回未写入/不一致的 (q, alpha)。原有相关性检查对 scale/zero 损坏免疫（仿射不变）。`_quant_dispatch` 改为 **(q, alpha) 重构原权重比对**全量校验，不过则 CPU 重量化。
3. **§17 嵌套 LoRA Linear 权重丢失**：非 FakeLinear 的嵌套 Linear（LoRA）以 external Convolution 形式存在，`rebuild()` 截断重写权重文件后引用失效、round-trip 物化为全 0。改为截断前内联保留。
4. **§18 transformers≥5 静默漏载 lm_head（本次用户实测乱码的根因）**：transformers 5.12 加载 RWKV7 时 untied 的 `lm_head.weight` 不被写入（残留未初始化内存），而 missing/mismatched/error 报告全为空 → logits 全坏 → 导出/推理乱码。`LlmModel.from_pretrained` 后新增 `_repair_output_embedding`：与 checkpoint 比对 absmax，偏差 >1% 时重新载入（支持单文件/分片 safetensors 与 pytorch bin，tied 权重跳过，正常加载时为 no-op）。

> 注：第一天曾误判 base 环境（fla 0.4.0）推理乱码为 fla 版本问题，后证实与导出乱码同为 §18 的 lm_head 加载 bug；修复后 base 环境（fla 0.4.0 + transformers 5.12.1）全流程正常。

## 5. 验证结果

| 验证项 | 结果 |
|---|---|
| HF 基线（greedy） | " A large language model (LLM) is a machine learning model..." |
| Python hook 对齐（embed/layer0/lastlayer/norm/logits × prefill/decode） | 全部 diff ≤ 1.3e-5，top1 token 精确一致（300 / 38864） |
| Python `--test` 贪心生成 | 与 HF 逐 token 一致 |
| C++ `llm_demo`（4bit HQQ，英文） | "Large language models (LLMs) are computer systems that can learn and understand language..." ✅ |
| C++ `llm_demo`（中文） | 连贯中文输出（0.1B 模型自身能力水平，与 HF 一致）✅ |
| lm_head 解量化校验 | max err 0.0134（正常 4bit HQQ 水平） |
| 性能（x86 CPU，0.1B 4bit） | prefill ~307 tok/s，decode ~128 tok/s |

## 6. 使用方法

```bash
# 导出（base 环境即可）
cd transformers/llm/export
python llmexport.py --path ~/data/models/RWKV7-Goose-World2.8-0.1B-HF --export mnn --hqq

# C++ 推理
cd build && cmake .. -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON && make -j$(nproc)
./llm_demo ../transformers/llm/export/model/config.json prompt.txt
```

提示：确定性贪心输出请在 `config.json` 设 `"sampler_type": "greedy"`；不要用 `"temperature": 0`（sampler 中 `1.0/temperature` 除零会产生乱码，既有行为）。

## 7. 改动文件

```
source/backend/cpu/CPULinearAttention.{cpp,hpp}     rwkv7_mixing / rwkv7 算子
transformers/llm/export/llmexport.py                tokenizer_file 修正
transformers/llm/export/utils/model_mapper.py       regist_rwkv7
transformers/llm/export/utils/transformers.py       RWKV7Attention / RWKV7Mlp / pre_norm
transformers/llm/export/utils/tokenizer.py          RWKV World 词表导出
transformers/llm/export/utils/model.py              _repair_output_embedding（§18）
transformers/llm/export/utils/mnn_converter.py      嵌套 external 权重内联（§17）
transformers/llm/export/utils/torch_utils.py        量化结果校验 + CPU 回落（§16）
skills/support-new-llm/SKILL.md                     已支持模型表新增 rwkv7
skills/support-new-llm/common-pitfalls.md           §15–18 经验沉淀
```

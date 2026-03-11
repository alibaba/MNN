# 步骤 6：特殊架构支持（新算子）

> **目标**：为全新架构的模型添加支持（如新的 Attention 类型、混合 conv/attention 架构等）。
>
> **前置条件**：步骤 1 判定为 Tier 6。
>
> **鼓励全栈实现**（Python 导出 + MNN Converter 解析 + C++ 后端算子）。如果遇到极度复杂的特殊硬件或底层系统问题，再考虑向人工求助。

---

## 6.1 识别新组件

从步骤 1 的分析中，明确需要新增的组件：

```
需要新增的组件类型：____（如 LinearAttention 变体、新的 conv 层、新的 MLP 等）
HF 源码中的类名：  ____
该组件替代的是：    ____（替代 self_attn / 替代 mlp / 全新位置）
该组件的输入/输出：  ____（通常是 [B, L, hidden_size] → [B, L, hidden_size]）
该组件是否有状态：  ____（如 conv state / recurrent state）
```

### Tier 6 的两大子类

| 子类 | 特征 | 典型模型 | 主要工作 |
|------|------|---------|---------|
| **混合架构** | `layer_types` 中有非 Attention 层（conv/mamba/rwkv），与 full_attention 交替 | lfm2 | 新 LinearAttention 变体 + C++ attn_type |
| **全新 Attention** | 所有层都使用非标准 Attention（如 gated delta rule） | qwen3_5 | 新 LinearAttention 变体 + C++ attn_type |

**共同点**：两者都通过 `FusedLinearAttention` 自定义算子导出，通过 `CPULinearAttention` C++ 类执行。区别仅在于是否与标准 Attention 层混合。

---

## 6.2 LinearAttention 架构参考

> **本节是 Tier 6 的核心知识**。大多数新架构最终都通过 LinearAttention 框架实现。

### 6.2.1 整体数据流

```
Python 侧（transformers.py）               ONNX 导出               C++ 侧（CPULinearAttention.cpp）
┌─────────────────────────┐                                        ┌─────────────────────────┐
│ ShortConvAttention      │                                        │ CPULinearAttention       │
│ / LinearAttention       │                                        │                         │
│                         │   FusedLinearAttention                  │  onResize():            │
│ __init__:               │   custom op (ONNX)                     │    分配 conv/rnn state  │
│   ModelMapper.do_map()  │ ──────────────────────>                 │    分配临时 buffer      │
│   FusedLinearAttention()│                                        │                         │
│                         │                                        │  onExecute():           │
│ forward (test path):    │                                        │    dispatch by          │
│   完整的 Python 计算    │                                        │    mAttentionType       │
│                         │                                        │    → short_conv()       │
│ forward (ONNX path):    │                                        │    → gated_delta_rule() │
│   调用 fused_attn op    │                                        │    → new_type()         │
└─────────────────────────┘                                        └─────────────────────────┘
```

### 6.2.2 FusedLinearAttention 自定义算子接口

**文件**：`transformers/llm/export/utils/custom_op.py`

```
算子名：LlmExporter::FusedLinearAttention

输入 Tensor（4 个，顺序固定）：
  [0] qkv          [B, D, L]   投影输出（conv 之前），D = 总维度
  [1] gate         [B, L, H]   decay / gate（不需要时传 zeros）
  [2] beta         [B, L, H]   learning rate（不需要时传 zeros）
  [3] conv_weight  [C, 1, K]   depthwise conv 权重，C = conv 通道数

属性（Attributes）：
  attn_type      string   算子子类型，如 "gated_delta_rule" / "short_conv"
  num_k_heads    int      K/Q 的 head 数（short_conv 可设 1）
  num_v_heads    int      V 的 head 数（short_conv 可设 1）
  head_k_dim     int      每个 K head 的维度
  head_v_dim     int      每个 V head 的维度
  use_qk_l2norm  int      是否对 Q/K 做 L2 归一化（0/1）

输出 Tensor（1 个）：
  [0] attn_out     [B, L, num_v_heads, head_v_dim]
```

**关键设计**：不同 `attn_type` 共享同一个算子接口，通过属性区分行为。这样 Converter 只需注册一次，新类型只需在 C++ Execution 中增加 dispatch 分支。

### 6.2.3 CPULinearAttention C++ 架构

**文件**：
- `source/backend/cpu/CPULinearAttention.hpp`
- `source/backend/cpu/CPULinearAttention.cpp`

#### 类结构

```cpp
struct StateCache {
    std::shared_ptr<Tensor> mConvState;      // Conv1D padding state
    std::shared_ptr<Tensor> mRecurrentState; // 递归状态（仅部分类型需要）
};

class CPULinearAttention : public Execution {
    // 算子参数（从 FlatBuffers 读取，对应 ONNX 属性）
    std::string mAttentionType;
    int mHeadKDim, mHeadVDim, mNumKHeads, mNumVHeads;
    bool mUseQKL2Norm;

    // 持久状态（通过 onClone 在 prefill/decode Execution 间共享）
    std::shared_ptr<StateCache> mStateCache;

    // 临时 buffer（每次 onResize 重新分配）
    std::shared_ptr<Tensor> mConvPadded, mConvOut;
    std::shared_ptr<Tensor> mTempVPred, mTempDelta;  // 仅 gated_delta_rule 需要
};
```

#### onResize 的统一 buffer 分配模式

```cpp
// ─── Per-type 参数（添加新类型只需在这里加分支）───
int convChannels = convDim;        // 默认：conv 覆盖所有通道
bool needRecurrentState = false;   // 默认：不需要递归状态

if (mAttentionType == "short_conv") {
    convChannels = mHeadVDim;      // conv 只覆盖 hidden_size 通道
} else if (mAttentionType == "gated_delta_rule") {
    needRecurrentState = true;     // 需要 [B, H, dk, dv] 递归状态
}
// 新类型在此添加 else if ...

// ─── 以下是共用逻辑，不需要修改 ───
// 1. 分配 conv state [B, convChannels, kernelSize-1]（STATIC，跨 decode 保持）
// 2. 如果 needRecurrentState，分配 recurrent state（STATIC）
// 3. 分配 mConvPadded / mConvOut 临时 buffer（DYNAMIC）
// 4. 如果 needRecurrentState，分配 mTempVPred / mTempDelta（DYNAMIC）
```

#### onExecute 的 dispatch

```cpp
ErrorCode onExecute(...) {
    if (mAttentionType == "short_conv") {
        short_conv(inputs, outputs);
    } else if (mAttentionType == "gated_delta_rule") {
        gated_delta_rule_mnn(inputs, outputs);
    }
    // 新类型在此添加 else if ...
    return NO_ERROR;
}
```

#### onClone 的状态共享

```cpp
bool onClone(Backend* bn, const Op* op, Execution** dst) {
    auto tmp = new CPULinearAttention(bn, op);
    tmp->mStateCache = mStateCache;  // 共享持久状态
    *dst = tmp;
    return true;
}
```

### 6.2.4 Python 侧 create_linear_attention 工厂

**文件**：`transformers/llm/export/utils/transformers.py`

```python
def create_linear_attention(attn, layer_id, config, rotary, mapper):
    """Factory function for creating LinearAttention variants based on config."""
    if hasattr(config, 'conv_L_cache') and config.conv_L_cache > 0:
        return ShortConvAttention(attn, layer_id, config, mapper)
    # 新类型在此添加 elif ...
    return LinearAttention(attn, layer_id, config, rotary, mapper)
```

`Decoder.__init__` 通过 `linear_attn` 槽位触发：

```python
# Decoder.__init__ 中的关键逻辑：
if hasattr(self, 'linear_attn') and self.linear_attn is not None:
    self.self_attn = create_linear_attention(self.linear_attn, layer_id, config, rotary, mapper)
    self.layer_type = 'linear_attention'
```

**映射侧**：在 `model_mapper.py` 的 decoder 映射中，将 HF 模型的非标准层映射到 `linear_attn` 槽位：

```python
decoder = {
    'self_attn': 'self_attn',         # 标准 Attention 层（有的层有，有的层没有）
    'linear_attn': 'conv',            # 非标准层 → linear_attn 槽位
    'mlp': 'feed_forward',
    # ...
}
```

对于混合架构（如 lfm2），同一个 decoder 映射同时包含 `self_attn` 和 `linear_attn`。`ModelMapper.do_map` 会对不存在的属性设置 `None`，所以：
- conv 层：`self_attn=None`, `linear_attn=convModule` → `ShortConvAttention`
- attention 层：`self_attn=attnModule`, `linear_attn=None` → `Attention`

---

## 6.3 添加新 LinearAttention 变体的 Checklist

> 以 `short_conv`（LFM2）为实际案例说明。

### Python 侧（4 个文件）

#### 1. `model_mapper.py` — 注册映射

```python
def regist_lfm2(self):
    # config 映射：添加模型特有的配置字段
    lfm2_config = {
        'hidden_size': 'hidden_size',
        # ...
        'conv_L_cache': 'conv_L_cache',  # ← 新字段
    }
    # linear_attention 映射：新组件的子模块名
    lfm2_linear_attention = {
        'in_proj': 'in_proj',
        'conv': 'conv',
        'out_proj': 'out_proj',
    }
    # decoder 映射：linear_attn 指向 HF 模型中的非标准层
    lfm2_decoder = {
        'self_attn': 'self_attn',
        'linear_attn': 'conv',         # ← HF 的 conv 层 → linear_attn 槽位
        'mlp': 'feed_forward',
        # ...
    }
    lfm2_map = {
        'config': lfm2_config,
        'model': lfm2_model,
        'decoder': lfm2_decoder,
        'attention': lfm2_attention,
        'linear_attention': lfm2_linear_attention,  # ← 新组件映射
    }
    self.regist('lfm2', lfm2_map)
```

#### 2. `config.py` — 注册新配置字段（如需要）

```python
self.conv_L_cache = kwargs.pop("conv_L_cache", 0)
```

#### 3. `transformers.py` — 新组件类 + 工厂注册

每个 LinearAttention 变体都是一个 `torch.nn.Module` 子类，需要实现两条路径：

```python
class ShortConvAttention(torch.nn.Module):
    def __init__(self, attn, layer_id, config, mapper):
        super().__init__()
        # 1. 用 ModelMapper.do_map 提取子模块
        ModelMapper.do_map(self, attn, mapper['linear_attention'])

        # 2. 创建 FusedLinearAttention 实例（用于 ONNX 导出）
        self.fused_attn = FusedLinearAttention(
            name=f'/layers.{layer_id}/self_attn/FusedLinearAttention',
            attn_type="short_conv",     # ← 新的 attn_type 字符串
            num_k_heads=1, num_v_heads=1,
            head_k_dim=self.hidden_size,
            head_v_dim=self.hidden_size,
            use_qk_l2norm=False
        )
        # 3. 初始化内部状态（用于 test path 的 decode 推理）
        self.conv_state = None

    def forward(self, hidden_states, attention_mask=None):
        if torch.onnx.is_in_onnx_export():
            # ONNX 路径：调用 FusedLinearAttention 自定义算子
            # 输入：投影后的 tensor [B, D, L]
            # 不需要的输入传 zeros
            attn_out = self.fused_attn(bcx_t, gate_zeros, beta_zeros, self.conv.weight)
            return self.out_proj(attn_out.view(B, L, -1))

        # Test 路径：完整的 Python 计算（用于 hook 对齐验证）
        # 对照 HF 源码实现完整的前向逻辑
        # 维护 self.conv_state 等内部状态
        ...
```

在 `create_linear_attention` 工厂中注册：

```python
def create_linear_attention(attn, layer_id, config, rotary, mapper):
    if hasattr(config, 'conv_L_cache') and config.conv_L_cache > 0:
        return ShortConvAttention(attn, layer_id, config, mapper)
    # elif some_other_condition:
    #     return NewTypeAttention(attn, layer_id, config, mapper)
    return LinearAttention(attn, layer_id, config, rotary, mapper)
```

#### 4. `custom_op.py` — 通常不需要修改

`FusedLinearAttention` 已支持任意 `attn_type` 字符串，新类型不需要修改 custom_op.py。

### C++ 侧（2 个文件）

#### 5. `CPULinearAttention.hpp` — 添加方法声明

```cpp
void short_conv(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
```

#### 6. `CPULinearAttention.cpp` — 3 处修改

**修改 1**：`onResize` 顶部的 per-type 参数分支

```cpp
if (mAttentionType == "short_conv") {
    convChannels = mHeadVDim;
} else if (mAttentionType == "gated_delta_rule") {
    needRecurrentState = true;
} else if (mAttentionType == "new_type") {
    // 设置 convChannels 和 needRecurrentState
}
```

**修改 2**：`onExecute` 的 dispatch 分支

```cpp
if (mAttentionType == "short_conv") {
    short_conv(inputs, outputs);
} else if (mAttentionType == "new_type") {
    new_type(inputs, outputs);
} else {
    gated_delta_rule_mnn(inputs, outputs);
}
```

**修改 3**：新方法实现

```cpp
void CPULinearAttention::short_conv(const std::vector<Tensor*>& inputs,
                                     const std::vector<Tensor*>& outputs) {
    // 输入：qkv [B, D, L], conv_weight [C, 1, K]
    // 输出：attn_out [B, L, num_v_heads, head_v_dim]
    //
    // 核心步骤：
    // 1. 从 qkv 中提取需要的分量
    // 2. 使用 mStateCache->mConvState 做 depthwise conv1d（带状态管理）
    // 3. 后处理（如元素乘法）
    // 4. 写入 output tensor
    //
    // 多线程：使用 MNN_CONCURRENCY_BEGIN/END，按 B*channels 并行
}
```

### 构建与测试

```bash
# 构建（只需要编译 LLM 相关目标）
cmake --build build --target llm_demo -j$(nproc)

# 导出
cd transformers/llm/export
python3 llmexport.py --path /path/to/model --export mnn --dst_path /tmp/MODEL

# C++ 推理测试
echo "你好" > /tmp/prompt.txt
./build/llm_demo /tmp/MODEL/llm_config.json /tmp/prompt.txt
```

---

## 6.4 已有 attn_type 实现参考

| attn_type | 模型 | conv 通道 | 递归状态 | SiLU | 核心逻辑 |
|-----------|------|----------|---------|------|---------|
| `gated_delta_rule` | qwen3_5 | D (全部) | 是 [B,H,dk,dv] | 是 | conv→SiLU→split QKV→L2norm→scale→delta rule recurrence |
| `short_conv` | lfm2 | H (部分) | 否 | 否 | split BCx→B*x→conv→C*conv_out |

### 新类型实现时的关键决策

1. **conv 覆盖多少通道？** → 决定 `convChannels` 值
2. **是否需要递归状态？** → 决定 `needRecurrentState`
3. **conv 后是否有激活函数？** → gated_delta_rule 有 SiLU，short_conv 没有
4. **qkv tensor 的语义是什么？** → 不同类型的 split 方式不同
5. **gate / beta 输入是否使用？** → short_conv 传 zeros，gated_delta_rule 实际使用

---

## 步骤 6 测试标准

### Python 侧通过标准

- [ ] 新组件类实现完成，test path 和 ONNX path 都有
- [ ] model_mapper.py 中新组件映射已添加（包含 `linear_attention` 子映射）
- [ ] config.py 中新配置字段已注册（如需要）
- [ ] create_linear_attention 工厂已注册新类型
- [ ] Python test 推理输出正确（与 HF 原始模型一致）
- [ ] ONNX 导出不报错

### C++ 侧通过标准

- [ ] CPULinearAttention.hpp 中新方法已声明
- [ ] onResize 的 per-type 参数已添加
- [ ] onExecute 的 dispatch 已添加
- [ ] 新方法实现完成（含 conv state 管理 + 多线程）
- [ ] C++ 推理输出正确

### 部分完成的评估机制

若全栈实现顺利，则直接报告完全成功。在极端情况下若底层算子实现卡住，允许分段交付：

```
✅ 已完成：
- Python 侧新组件实现（test path + ONNX path）
- model_mapper.py 映射
- Python test 验证通过
- ONNX 导出成功

⏳ 待解决：
- CPULinearAttention 中的新 attn_type 实现
```

---

## 下一步

- **Python 侧完成后** → 回到 `step3-test-python.md` 重新验证
- **全部完成后** → 回到 `step4-export.md` 完成最终导出验证
- **C++ 侧无法完成** → 总结工作，请求人工协助

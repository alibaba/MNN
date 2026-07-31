# Hexagon HVX FP16 PWL 激活函数

## 背景与目标

Hexagon 后端使用分段线性近似（Piecewise Linear，PWL）实现部分 FP16 激活函数，从而避免在向量路径中调用
`exp`、`tanh` 等标量超越函数。对于每个输入 lane，PWL 内核先选择输入所在的分段，再计算：

```text
y = a[segment] * x + b[segment]
```

该实现并非只追求减少数学分段数，而是针对 HVX FP16 算术和 `vlut16` 指令共同设计。分段边界、FP16
系数量化、查找表排布和分段索引生成开销需要一起评估。

当前优化覆盖以下算子：

| 算子 | 默认实现 |
| --- | --- |
| Sigmoid | 16 段压缩式 PWL |
| Tanh | 12 段压缩式 PWL |
| GELU | 12 段压缩式 PWL |
| SiLU | 面向 HVX 指令约束学习得到的 8 段 PWL |
| MulSiLU | 复用 learned8 SiLU，随后执行 FP16 乘法 |
| Log | HVX `log2` 乘以 `ln(2)`，不使用 PWL |

未填满一个 HVX 向量的尾部元素仍使用标量实现，以保证任意输入长度下的正确性。

## 编译变体

编译参数 `HTP_OPS_PWL_VARIANT` 用于选择具体实现：

| 参数值 | SiLU/MulSiLU | 其他 PWL 激活函数 | 用途 |
| --- | --- | --- | --- |
| `uniform32` | `[0, 8]` 上的 32 个等宽分段 | 等宽分段表 | 精度和性能对照基线 |
| `companded16` | 16 个非均匀分段 | 单表、最多 16 个分段 | 较小的通用实现 |
| `learned8` | 学习得到的 8 段实现 | 与 `companded16` 相同 | 默认配置 |

进入 HTP 算子库目录：

```sh
cd source/backend/hexagon/htp-ops-lib
```

直接使用 SDK 编译时可以执行：

```sh
build_cmake hexagon DSP_ARCH=v79 HTP_OPS_PWL_VARIANT=learned8
```

项目构建脚本也支持通过环境变量选择变体：

```sh
# 默认使用 learned8
bash build.sh v79

# 编译对照变体
HTP_OPS_PWL_VARIANT=companded16 bash build.sh v79
```

CMake cache 会保留之前的配置。在已有构建目录中切换变体时，应使用干净的构建目录，或者显式传入
`HTP_OPS_PWL_VARIANT`。

## learned8 SiLU 设计

默认 SiLU 近似使用以下绝对值区间：

```text
[0, 0.25), [0.25, 0.5), [0.5, 1), [1, 1.5),
[1.5, 3.5), [3.5, 5), [5, 6), [6, 8)
```

对于每个 HVX FP16 输入向量，快速路径执行：

1. 提取符号位和 FP16 绝对值位模式。
2. 将指数和尾数高位压缩成 16 种状态。
3. 使用一次 `vlut16` 将状态映射到 8 个分段之一。
4. 再使用两次 `vlut16` 分别读取 FP16 斜率 `a` 和偏置 `b`。
5. 使用 QF16 乘加计算 PWL 结果。
6. 利用 `SiLU(-x) = SiLU(x) - x` 恢复负半轴结果。
7. 当 `|x| >= 8` 时，正半轴饱和到 `x`，负半轴饱和到零。

状态到分段的映射表为：

```text
0, 0, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5, 5, 6, 7, 7
```

虽然 learned8 只需要 8 对系数，但 HVX 查找表仍需要按照向量布局进行 128 字节对齐和填充。因此，数学
分段更少并不代表最终 DSP skeleton 一定更小。

## 系数生成与 CPU 模拟

[`tools/pwl_search.py`](../../source/backend/hexagon/htp-ops-lib/tools/pwl_search.py) 是 CPU 参考实现和系数生成
工具。learned8 模拟器会覆盖：

- 所有有限 FP16 输入；
- 真机测试中的 FP32 到 FP16 输入转换，同时保留 FP32 参考结果；
- 斜率和偏置的 FP16 量化；
- QF16 计算结果转换回 FP16 时的舍入；
- 与 DSP 内核一致的 FP16 位状态编码器。

检查默认 SiLU 实现的精度：

```sh
python3 tools/pwl_search.py --variant learned8 --function silu --check
```

检查通用对照变体：

```sh
python3 tools/pwl_search.py --variant companded16 --function all --check
python3 tools/pwl_search.py --variant uniform --function all --check
```

增加 `--emit-c` 参数可以输出生成的 FP16 系数表。

## 代码结构

- [`include/dsp/pwl.h`](../../source/backend/hexagon/htp-ops-lib/include/dsp/pwl.h)：HVX 分段索引、查表、PWL
  计算、对称关系和饱和处理。
- [`src/dsp/pwl.cc`](../../source/backend/hexagon/htp-ops-lib/src/dsp/pwl.cc)：对齐后的系数表和索引表。
- [`src/dsp/unary_ops.cc`](../../source/backend/hexagon/htp-ops-lib/src/dsp/unary_ops.cc)：Sigmoid、Tanh、
  GELU、SiLU 和 Log 向量路径。
- [`src/dsp/eltwise_ops.cc`](../../source/backend/hexagon/htp-ops-lib/src/dsp/eltwise_ops.cc)：Binary `MulSiLU`。
- [`src/dsp/loop_ops.cc`](../../source/backend/hexagon/htp-ops-lib/src/dsp/loop_ops.cc)：Loop 内部的 `MulSiLU`。

`vlut16` 会分别消费输入 halfword 的高、低两个字节。实现中将 4 bit 分段索引复制到两个字节，再读取
lookup 结果的低向量，从而保留全部 64 个 FP16 lane。修改索引编码后，必须在 DSP 上使用各 lane 不同的
输入验证映射关系。

## 精度测试

后端专项测试包括：

- [`HexagonUnaryPWLTest.cpp`](../../test/op/HexagonUnaryPWLTest.cpp)
- [`HexagonMulSiluPWLTest.cpp`](../../test/op/HexagonMulSiluPWLTest.cpp)

当运行时没有选择 Hexagon 后端时，这两个测试会自动跳过。在启用 Hexagon 的 Android 构建中可以执行：

```sh
./run_test.out op/hexagon/unary-pwl 10 2 1
./run_test.out op/hexagon/mul-silu-pwl 10 2 1
```

learned8 在一台 v79 真机上的验证结果如下：

| 算子 | 最大绝对误差 | 测试阈值 |
| --- | ---: | ---: |
| Sigmoid | 0.00236678 | 0.005 |
| Tanh | 0.00627482 | 0.009 |
| SiLU | 0.00754023 | 0.008 |
| GELU | 0.00613671 | 0.009 |
| Log | 0.00332212 | 0.02 |
| MulSiLU | 0.07119751 | 0.08 |

对于 learned8 SiLU，遍历所有有限 FP16 输入时的最大绝对误差为 `0.00632850`；使用 FP32 测试输入并经过
FP16 转换后的最大误差为 `0.00722693`。后者更接近实际 Host 到 DSP 的输入路径。

## 性能结果

参考 v79 设备上使用相同 Host/runtime/test tuple 的测试结果如下。PWL 前原始实现取自提交
`9cb231e23b`，测试时仅替换 DSP skeleton；表中为 DSP 耗时中位数：

| 测试项 | PWL 前原始实现 | companded16 | learned8 | learned8 相对 PWL 前耗时降低 | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MulSiLU 单算子（262144 个元素） | 10.2770 ms | 5.2995 ms | 4.8850 ms | 52.47% | 2.10x |
| Qwen3-0.6B `BINARY_ELEMENTWISE` prefill | 56.0425 ms | 34.7880 ms | 32.0605 ms | 42.79% | 1.75x |
| Qwen3-0.6B `BINARY_ELEMENTWISE` decode | 49.3120 ms | 37.1125 ms | 35.7615 ms | 27.48% | 1.38x |

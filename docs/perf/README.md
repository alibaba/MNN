# MNN 性能优化专区

本目录汇总 MNN 在不同硬件 / 算子上的性能优化实现说明,作为 review、扩展与移植的参考。

## 索引

| 文档 | 内容 |
|------|------|
| [`arm_low_bit_gemm.md`](./arm_low_bit_gemm.md) | ARM CPU 低 bit (W2 / W3 / W4 / W8) GEMM kernel 数据排布与汇编优化 |

> 后续新增:LinearAttention CPU 优化、TurboQuant KV Cache、CUDA Blackwell sm_120 适配 等,欢迎按相同结构补充。

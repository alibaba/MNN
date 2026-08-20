# Kernel 类优化手段

单个 kernel 内部的计算/访存形态改造。每条含：原理、适用条件、陷阱、验证方式。

## 1. GEMV 融合 epilogue（尾段折叠）

- **原理**：decode 主体是带宽瓶颈型 GEMV。把紧随其后的逐元素算子
  （SwiGLU/bias/激活）折进 GEMV 尾段就地计算，省一次 dispatch 的固定开销，
  更省中间结果写回再读回的显存往返。
- **适用**：尾段只依赖本 kernel 已算出的元素；尾段算子逐元素或短邻域。
- **陷阱**：需要多路结果对齐的尾段（如 SwiGLU 需 gate/up 同 TG）要先解决
  数据汇聚问题（见算子级 gate/up 合并）。
- **验证**：对拍融合前后输出 bit 级一致；确认 dispatch 数真的减少。

## 2. LN 前序拆分到多 simdgroup

- **原理**：多 SG kernel 中若每个 SG 各自加载同一份输入做前处理，读取量按
  SG 数翻倍；改为按 SG 分工 + threadgroup 内存交换，公共输入只读一次。
- **适用**：2sg 及以上、前处理输入相同的 GEMV/GEMM kernel。
- **陷阱**：**部分线程提前退出 + barrier 是 UB**——必须全部线程到达 barrier
  后再分工，不能靠早退省工作。
- **验证**：输入读取量（profile 字节数）减半；输出对拍。

## 3. Split-K GEMV 及变体

- **原理**：小 batch GEMV 并行度不足时沿 K 切分给更多 lane/SG。收益本质是
  **翻倍在途 lane、提高访存并发**，不是省掉 barrier。
- **变体**：双 SG + tg 内存归并（SPLIT_K_2）；单 SG 内累加 + simd shuffle
  收拢（免 tg 内存/barrier，TG 可缩小）。
- **适用**：K 很大而并行单元不饱和的 GEMV（decode 短上下文）。
- **陷阱**：K 切分使 weight 读取模式改变，量化块边界要与切分对齐，否则
  跳块/错块（lane 拆分 bug 曾跳半数 weight 块）；若该 kernel 后紧跟
  GPU→CPU 同步，kernel 加速不兑现为 e2e。
- **验证**：conv/wquant 单测全变体通过；e2e 配对 A/B 而非只看 kernel 计时。

## 4. 向量宽 load 与访存合并

- **原理**：带宽瓶颈 kernel 的快慢取决于访存模式能否吃满 DRAM 带宽。
  lane 持连续若干元素一条向量 load（ftype4/char4），simdgroup 32 lane
  恰好覆盖整行 → 完全合并 burst；反之逐 token 跨步标量读每 2KB 只碰几字节，
  load 指令数翻数倍、burst 利用率骤降。
- **适用**：一切流式读取 KV cache / 权重的 kernel。
- **要点**：**数据布局决定 kernel 可达的访存形态**。想让某 kernel 跑成合并
  读，先改布局（布局是全局不变量，翻转必须所有读写方原子落地）。
- **验证**：profile 带宽兑现率；同字节数下 load 指令数对比。

## 5. 寄存器驻留 + 单遍流式（sdpa_vector 形态）

- **原理**：decode attention（seq_q=1）最优形态：Q 进寄存器不再碰显存；
  逐 token 交错流式（`i = sgitg; i < kv; i += NSG`），每 token K 行点积 →
  simd_sum → 在线 softmax（M/S）→ 立即用同 token V 行更新 O；score 不落
  任何内存，无第二段 AV dispatch；跨 SG 归并用转置写法让归并读合并。
- **适用**：decode 单 query 的融合 attention。
- **陷阱**：**lane↔输出维度映射**。流式循环 lane 持 `d = lane*DPT + dd`，
  归并写回若沿用旧映射会短 KV 正常、长 KV 后乱码。对拍必须覆盖长 KV +
  prefill 后首个 decode token。
- **验证**：与禁用路径（env 开关）greedy 逐字节对拍，长 prompt 必测。

## 6. 量化解包向量化与公共加载去重

- **原理**：int4/int8 解包用向量指令一次 4/16 元素；多 SG 共用输入只加载
  一次经 threadgroup 共享。
- **陷阱**：把向量积（in4 × FLOAT4x4）重构成标量循环极易写成**转置乘积**
  且 scale/bias lane 错位——此类重构必须 bit 级对拍，肉眼/greedy 短 prompt
  都可能漏掉。
- **验证**：fp32 bit-identical 或量化单测全模式通过。

## 7. 编译期常量与 host 预算

- **原理**：host 可确定的量（split-K 中段步长等）以宏注入编译期，省 kernel
  内除法/分支。
- **陷阱**：甄别伪优化——循环边界常量化若寄存器压力不变可能零收益；
  先量化收益来源再投入。
- **验证**：汇编/寄存器占用对比 + 配对 A/B。

## 8. 线程组规模（NSG）校准

- **原理**：单 workgroup kernel 的 simdgroup 数是占用率/调度开销/归并成本
  的三方折中，无跨设备通用最优。
- **要点**：按设备档（tensor-API 与否）分别 sweep；候选值全集都要测
  （只比两端会漏中间最优点）；KV/prompt 维度也要扫（最优点会移动）。
- **验证**：多轮配对、逐对同向性检查，而非单次均值。

## 9. 递推状态驻留与并行扫描（LinearAttention）

- **原理**：递推/scan kernel 把状态从 device 往返改寄存器驻留（每步一次
  load/write）；chunk 内前缀和用 Hillis-Steele 并行 scan，前代求解摊到
  全部 simdgroup；窄 head_dim 写专用特化避免通用路径分支。
- **适用**：gated delta rule 等线性注意力的 chunk 递推。
- **验证**：与参考实现长序列对拍（递推误差会随序列累积）。

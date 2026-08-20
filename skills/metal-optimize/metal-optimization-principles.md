# Metal LLM 推理优化方法原理(feature/metal-speed 战役沉淀)

本文只讲**优化原理与适用条件**,不含具体性能数字(数字见
`feature-metal-speed-perf.md`;可操作的检查清单见 `metal-perf-methods/`)。

优化手段按三个层次组织:

1. [Kernel 类](#一kernel-类)——单个 kernel 内部的计算/访存形态改造
2. [算子级别](#二算子级别)——跨算子的融合与图结构改造
3. [调度类](#三调度类)——CPU/GPU 协同、dispatch 与同步路径

---

## 一、Kernel 类

### 1.1 GEMV 融合 epilogue(尾段折叠)

decode 的主体是 GEMV(权重带宽瓶颈)。把紧随 GEMV 的逐元素算子
(SwiGLU、bias、激活)折进 GEMV kernel 的尾段,在结果写回前就地计算:

- 省掉一次独立 dispatch 的固定开销;
- 省掉中间结果写显存、再读回来的往返(对带宽瓶颈 kernel,往返本身就是大头);
- 前提:尾段计算只依赖本 kernel 已算出的元素(SwiGLU 需要 gate/up 两路对齐
  到同一 threadgroup,见算子级 §2.1)。

### 1.2 LN 前序拆分到多个 simdgroup

2-simdgroup(2sg)GEMV kernel 中,每个 SG 各自加载 input+residual 做 LN 前序,
读取量翻倍。把前序按 SG 拆分、经 threadgroup 内存交换,输入读取减半。
注意:任何"部分线程提前退出 + barrier"的组合都是 UB(陷阱:提前退出的线程
到不了 barrier),必须让全部线程到达 barrier 后再分工。

### 1.3 Split-K GEMV 及其变体

小 batch GEMV 的瓶颈是并行度不足(K 维串行)。沿 K 切分给更多 lane/SG:

- **SPLIT_K_2**:两个 SG 各算半个 K,threadgroup 内存归并。收益本质是
  **翻倍在途 lane**,提高访存并发;
- **SPLIT_K_SHUFFLE**:同一 SG 内累加两半 K,用 simd shuffle 收拢,免
  threadgroup 内存与 barrier、TG 缩到 64 线程。实测证明其收益同样来自
  在途 lane 而非"免 barrier"——免 barrier 本身不值钱;
- **lm_head split-K 的教训**:kernel 再快,若该 kernel 之后紧跟 GPU→CPU
  同步(采样),e2e 不兑现。**优化前先确认瓶颈段是不是你在优化的那段。**

### 1.4 向量宽 load 与访存合并

带宽瓶颈 kernel 的快慢取决于访存模式能否吃满 DRAM 带宽:

- lane 持有**连续**若干元素、一条向量 load(`ftype4`/`char4`)完成,
  simdgroup 内 32 lane 恰好覆盖一整行 → 完全合并的 burst;
- 反例:逐 token 跨步标量读,每 2KB 只碰几个字节,load 指令数 ×4,
  burst 利用率低,带宽兑现率大幅下降;
- **数据布局决定 kernel 形态**:V cache 行主序(镜像 K)让"逐 token 流式、
  行内向量读"成为可能,这是 decode_splitkv 对齐 MLX sdpa_vector 的前提。
  布局是全局不变量,翻转必须所有读写方原子落地。

### 1.5 寄存器驻留与单遍流式(MLX sdpa_vector 形态)

decode attention(seq_len=1)的最优形态:

- Q 加载进寄存器一次,不再碰显存;
- 逐 token 交错流式(`i = sgitg; i < kv; i += NSG`),每 token:K 行做点积 →
  simd_sum 得 score → 在线 softmax 更新 M/S → **立即**用同 token 的 V 行
  更新 O;
- score 不落 threadgroup/全局内存,无第二段 AV dispatch;
- 跨 simdgroup 归并用转置写法(`s_out[lane*NSG + sg]`),让归并读合并。
- 易错点:**lane↔输出维度的映射**。流式循环里 lane 持有
  `d = lane*DPT + dd`,归并写回若沿用旧映射 `d = dd*32 + lane` 会产生
  "短 prompt 正常、长 prompt 后乱码"的隐蔽错误。正确性对拍必须覆盖长 KV。

### 1.6 量化解包向量化与公共加载去重

- 权重解包(int4/int8 → fp)用向量指令一次处理 4/16 元素,而非标量循环;
- 多 SG 共用的输入(如 LN 结果)只加载一次,threadgroup 内共享;
- 陷阱:把 `in4 * FLOAT4x4` 这类向量积重构成标量循环时,容易写成
  **转置乘积**且 scale/bias lane 错位——重构必须 bit 级对拍。

### 1.7 编译期常量与 host 预算

把 host 可确定的量(如 split-K 中段步长)以宏注入编译期,省掉 kernel 内
除法/分支。注意甄别伪优化:编译期常量化循环边界若寄存器压力不变,
收益可能为零(证伪案例:GEMV BLOCK_SLICES)。

### 1.8 线程组规模(NSG)校准

单 workgroup kernel(如融合 SDPA)的 simdgroup 数是占用率、调度开销、
归并成本的三方折中,**没有跨设备通用最优值**:

- 必须按设备档(tensor-API 与否)分别 sweep;
- sweep 要覆盖候选值全集(只比 8 vs 32 会漏掉 16 这个最优点);
- sweep 要覆盖实际 KV 分布(最优点可能随 KV 长度移动)。

### 1.9 递推型 kernel 的状态驻留与并行扫描(LinearAttention)

递推/scan 类 kernel(gated delta rule 等):

- 递推状态从 device 往返改为**寄存器驻留**,每步一次 load/write;
- chunk 内前缀和用 Hillis-Steele 并行 scan,前代求解摊到全部 simdgroup;
- 对窄 head_dim 写专用特化(如 dk==64),避开通用路径的分支。

---

## 二、算子级别

### 2.1 gate/up + SwiGLU 合并折叠

MLP 的 gate、up 两个投影结构相同、输入相同,合并为一次 GEMV 并在尾段做
SwiGLU(见 §1.1)。落地细节:

- 合并改变输出张量布局,下游 slice 要同步;
- 不同设备档的 pipeline 编译路径可能让融合"静默不命中",需要强制编译
  变体(如 ROW_2)兜底,并以 dispatch 日志验证真的走了融合路径。

### 2.2 QKV 打包 grid

Q/K/V 三个投影融合后,若按矩形 grid 派发,尺寸不齐时大量 threadgroup
早退空跑(混合架构的线性层可达 2/3 闲置)。把三段工作打包成一维紧凑
grid,消掉空 TG。适用一切"多段同质工作拼矩形 grid"的融合 kernel。

### 2.3 导出期声明的算子融合链路

MNN 的融合不在后端做图匹配,而是**导出期声明分组 → converter 吸收
(如 LN 吸收进 FusedLinear)→ 后端 setupFusion 装配 leader/follower**:

- 融合命中与否在导出阶段就决定,排查从导出配置开始;
- 同一结构要**无条件融合**(如 LinearAttention 的投影曾受 qkv-fusion
  开关牵连,关闭时产出错误图)——融合开关只应控制"快慢路径",
  不应产生语义不等价的图;
- 铁律:把前驱折进后继前必查**内存别名**(前驱输入可能已被分配器
  复用为后继输出)。

### 2.4 融合路径的正确性门槛

融合改变了浮点累加顺序与精度路径,验收标准:fp32 bit-identical 最强,
fp16 greedy 逐字节对拍次之;**token 级"看起来一致"不算数**。

---

## 三、调度类

### 3.1 采样移出 expr/挪到设备侧,消除隐性 CPU 段

greedy ArgMax 若走通用 Express 接口,可能实际跑在 CPU executor 上,
成为每 token 数百微秒的隐性开销。原则:

- 采样是每 token 必经路径,用原生 SIMD 循环或设备端 kernel 实现;
- 警惕计时盲区:decode 计时若在采样之后启动,decode 速度指标对采样
  开销天然失明——优化采样前后要对比**整 token 周期**,而不是 decode 速度。

### 3.2 GPU→CPU 同步点即瓶颈点

decode 尾部(lm_head → logits → 采样)是天然的 GPU→CPU 同步点。
同步点之前的 GPU 加速若不能消除同步本身,e2e 收益会被压缩
(lm_head split-K 证伪即此)。调度类优化的第一优先级是**减少同步次数/
提前 overlap**,而不是加速同步点前的 kernel。

### 3.3 Encode Replay 与资源生命周期

重复 decode 帧用录制好的 command buffer 重放,省 CPU 编码。前提:
kernel 绑定的所有 buffer 地址在重放期间稳定——const buffer 在 resize
时**原地更新内容而非重建对象**,否则重放引用悬垂地址。新 op 接入
replay 前先审其 resize 行为。

### 3.4 路径选择的自动阈值与降级链

同一算子多条 kernel 路径时(如 decode attention 的 splitkv / 融合
qk_softmax / 三段式),用**可实测校准的阈值**自动路由:

- 阈值来自各路径的 A/B 交叉点,随 kernel 演进必须重新校准
  (V 布局翻转后回退路径变慢,交叉点大幅前移,阈值从 3072 降到 128);
- 保留 env 覆盖(禁用/自定义阈值)作为调试与对拍通道;
- 阈值决策的设备分档(tensor-API 与否)要显式,不假设跨档一致。

### 3.5 实验开关的收敛纪律

新优化先以默认关的 env 开关落地,充分 A/B 后转正为默认行为并**退役
开关**(避免开关组合爆炸)。证伪的方向连开关一起删除,只留归档记录。

---

## 方法学(贯穿三层)

1. **正确性先于性能**:先过 bit/greedy 对拍再看数字。错误 kernel 也能
   "变快"(曾出现乱码版本测出幻影 +13%)。
2. **A/B 必须交替配对**:热漂移可让同一二进制前后跑出 10%+ 差异;
   顺序单向对比不可信,必要时取每轮配对值分析。
3. **先定位瓶颈段再动手**:GPU busy vs wall、kernel 计时 vs e2e、
   带宽兑现率,三者口径不同,优化必须对准真正的瓶颈段。
4. **证伪也是成果**:中性/负收益的实验连同测试方法一起归档,
   防止重复投入。

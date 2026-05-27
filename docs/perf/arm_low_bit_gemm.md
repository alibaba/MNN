# ARM CPU 低 bit GEMM Kernel (W2 / W3 / W4 / W8)

> 范围: ARMv8.2 **sdot** (`SRC_UNIT=4`) 与 ARMv8.6 **i8mm / smmla** (`SRC_UNIT=8`),FP32 与 FP16 输出路径,共 W4/W8 老路径 + 本次新增的 W2/W3 路径。
>
> W2/W3 来自 commit `85e3f63a252 [CPU:Feature] Support low-bit ARM quant kernels`;W4/W8 早已上线,本文一并汇总以便对比。

---

## 0. 总览

低 bit GEMM 把每个 OC 的权重压缩到 ≤ 1 byte,反量化与计算融合在汇编里完成。统一约定:

* 计算单元:`UNIT = 8` 路 OC × `SRC_UNIT` 路 IC,即 sdot 一个 cell 32 weight、smmla 一个 cell 64 weight。
* 权重在 cell 内 **同 IC、跨 OC pack** 进单 byte;OC 间隔取 2 是为了让单条 `ushr/and` 直接吐出 smmla 期望的 OC pair。
* Kernel 输出 **unsigned [0, 2^bits − 1]**;`_computeReorderQuantInfo` 写入 `originOffset = -(2^bits / 2)`,在 post‑process 用一次 `MLA_WEIGHTZERO` 把 zero point 重定心到 signed,kernel 内部不出现 sub / 符号扩展(W3 unsigned-domain 是关键)。
* Post‑process 链统一为 `Int32ToFloat → MUL_SCALE → MLA_WEIGHTZERO → MUL_INPUT_SCALE → ADD_BIAS → ReLU`,FP16 路径多一组 `fcvtn / fcvtl`。
* W4/W8 成熟阶段已把 per-block dequant 链最末的 `fadd v_acc, v_per_block` 融进前面的链(`fmla v_acc, v_per_block, scale_eff`),W2/W3 直接复用同一融合后处理。

| 位宽 | weightBytes (有效) | originOffset | smmla 单 cell 字节 | sdot 单 cell 字节 | 解包关键宏 |
|------|--------------------|--------------|----|----|----|
| W8 | 1.00 | 0 | 64 | 32 | 不需要解包 |
| W4 | 0.50 | -8 | 32 | 16 | `ushr/and #15` |
| W3 | 0.375(sdot 实占 0.5,4B pad) | -4 | 24 (16 main + 8 aux) | 12 (8 main + 4 aux,padded 16) | `W3_UNPACK_SERIAL` / `UNPACK_W3_SDOT` |
| W2 | 0.25 | -2 | 16 | 8 | `ushr/ushl/and #3` / `UNPACK_W2_SDOT` |

---

## 1. 数据排布

### 1.1 W8 — int8 直存

* 一个 byte 一个 weight,`reorderWeight` 直接打成 `(cellCount, UNIT × SRC_UNIT)` 排布,smmla cell 64 B,sdot cell 32 B。
* signed int8;无需 `originOffset`。

### 1.2 W4 — 单 byte 2 路 OC

```
W4 byte = | bits[7:4] = OC 偶号  | bits[3:0] = OC 奇号 |
```

* **smmla cell (32 B)**:`byte i (0..15)` → IC = i,OC 偶号;`byte i (16..31)` → IC = i-16,OC 奇号。
* **sdot cell (16 B)**:`byte i (0..7)` → IC = i,OC 偶号;`byte i (8..15)` → IC = i-8,OC 奇号。
* 解包仅 `ushr v,#4` + `and v,v_mask15` 两条指令,产出两个 OC pair。

### 1.3 W2 — 单 byte 4 路 OC

```
W2 byte:  bits[7:6]=oc01  bits[5:4]=oc23  bits[3:2]=oc45  bits[1:0]=oc67
```

* **smmla cell (16 B)**:`byte 0..7` → IC=0..7, OC=0,2,4,6;`byte 8..15` → IC=0..7, OC=1,3,5,7。
* **sdot cell (8 B)**:`byte 0..3` → IC=0..3, OC=0,1,2,3;`byte 4..7` → IC=0..3, OC=4,5,6,7。
* OC 走 2-step 间隔是为了让 smmla 的 `ushr v0,#6 / #4 / #2 / and v0,#3` 4 步 ALU 直接拿到 4 个 OC pair。

### 1.4 W3 — bit-plane 2 + 1 拆分

W3 不能 4 路并打,改成位平面拆:**低 2 bit 与 W2 同 layout 存为 main**,**高 1 bit 单独存 aux**。

* **smmla cell (24 B)** = 16 B main + 8 B aux
  * aux IC-major:`byte i (i=0..7)` 存 8 个 OC 在 IC=i 的高 bit,bit `(7-j)` 表示 OC `j`。
  * 选 IC-major:i8mm 用 `ld1r {v1.2d},[x2],#8` 复制成 16B 后,通过 `ushl + and` 链 4 次抽出 4 个 OC pair 高 bit。
* **sdot cell (12 B + 4 B pad → 16 B)** = 8 B main + 4 B aux + 4 B 0-pad
  * aux OC-major:`byte k` 存 OC `k` 在 IC=0..3 的高 bit;0-pad 由 reorder 的 memset 给出。
  * `shape[4]` 在 `ConvInt8TiledExecutor.cpp` 向上取整到 2,否则 `shape[4]=1` 让 packer/kernel 越界写。

### 1.5 originOffset 语义

C++ 把 `wf = alpha * (wi_unsigned + originOffset) + zero` 中的 `originOffset` 注入每行 weight 的 zero point。kernel 输出 unsigned 域 → post‑process 用一次 `MLA_WEIGHTZERO` 把 zp 项打入,等价于把 unsigned 域中心化为 signed,**kernel 不再需要 sub / bic**。这是 W3 实测从 41.3 → 50.2 GB/s 的直接原因。

---

## 2. 汇编实现要点

### 2.1 smmla (i8mm) — TILE_10 共享骨架

```
LoopSz_TILE_10:
    ld1 {v0/...},[x2],#W            ; weight
    ld1 {v3..v6},[x11],#64          ; src tile 0..7
    ld1 {v7},[x11],#16              ; src tile 8..9
    <UNPACK 产出 v8..v11 = 4 个 OC pair>
    smmla v12,v3,v8 ... smmla v31,v7,v11  ; 5 行 src × 4 列 OC pair = 20 条 smmla
    bne LoopSz_TILE_10
```

* 累加寄存器恰好用满 `v12..v31`(20 个)。
* W4/W2/W3 的差异仅体现在 `<UNPACK>` 段,后处理(`ADD_BIAS_FLOAT/MUL_SCALE/...`)逐宏复用。

各 unpack 成本(per cell):

| 位宽 | 解包指令 | 备注 |
|------|----------|------|
| W4 | `ushr v8,v0,#4 ; and v10,v0,v2` × 2 vec → 4 ALU | 2 个 16B weight 寄存器 |
| W2 | `ushr v0,#6/#4/#2 ; and v9/v10/v11,v2(=#3)` → 6 ALU | 1 个 16B weight 寄存器 |
| W3 | W2 的 6 ALU + `W3_UNPACK_SERIAL` 中 aux 4× `ushl/add/and` ≈ 14 ALU | 主 16B + aux 8B(`ld1r`) |

### 2.2 sdot — TILE_1 主战场,多 tile fall-through

```
L8LoopSz_TILE_1_lu1:
    ld1 {v_w*},[x2],#cellBytes
    <UNPACK> -> v3 (OC0..3 ×4 IC), v12 (OC4..7 ×4 IC)
    ld1r {v0.4s},[x1],#stride    ; 1 个 input 4-byte
    sdot v8.4s, v3.16b,  v0.4b[0]
    sdot v9.4s, v12.16b, v0.4b[0]
    bne L8LoopSz_TILE_1_lu1
```

* W4 path:`UNPACK = ushr/and` 极简,TILE_12/8/4 都各自有完整 schedule。
* W2/W3 sdot:目前仅 TILE_1 有专门 schedule,TILE_12/8/4 fall-through 到 TILE_1 单 batch 循环以保正确;后续优化按 W4 模板补全。
* W2 sdot UNPACK(`UNPACK_W2_SDOT`)9 条:`tbl/ext/tbl/ushl/ushl/movi #3/and/and`。
* W3 sdot UNPACK(`UNPACK_W3_SDOT`):main 8 条(同 W2)+ aux 8 条(对称的 `tbl/ext/ushl(<<2)/and(#4)`)+ 2 条 `orr` 合并 = 18 条,每 cell 32 weight。
* `lu4` 模式预取 4 cell + 1 src,做 8 条 sdot,显著减少 IC 循环开销;tail 走 `lu1`。

### 2.3 W3 `W3_UNPACK_SERIAL` 详解

```asm
; aux_dup 已是 ld1r 复制后的 16B; main 是 16B
ushl oc01, aux_dup, shifts            ; shifts = {-7,-6,-5,-4,-3,-2,-1,0, ...}
add  t,    shifts,  idx (=2)
ushl oc23, aux_dup, t
add  t,    t,       idx
ushl oc45, aux_dup, t
add  t,    t,       idx
ushl oc67, aux_dup, t                 ; 4 个 OC pair 的高 bit 落在 bit2 位置
and  oc**, oc**, mask1 (=#4)          ; 只留 bit2

ushr t, main, #6 ; add oc01, oc01, t
ushr t, main, #4 ; and t, t, mask3 (=#3) ; add oc23, ...
ushr t, main, #2 ; and t, t, mask3       ; add oc45, ...
and  t, main, mask3                       ; add oc67, ...
```

* 4 次 `ushl/add` 让 aux 不依赖 tbl,与 main 的 ushr 链可乱序发射;mask3=#3、mask1=#4、idx=#2、shifts 全部是 cell 外预加载常量,inner loop 不再额外 movi。

### 2.4 与 W4 共用的循环外结构

* `REVERT_INPUT_DEQUANT_BIAS` / `REVERT_WEIGHT_KERNEL_SUM`:block 切换时把 src/weight kernel sum 指针回滚到当前 block 的起点。
* 多 block 复用 src,通过外层 `TILE10_BLOCKNUM` 循环;block 内 `LoopSz` 是 IC 循环。
* `LoopDz8 / LoopDz4`:tail OC 不足 8 时降到 4 OC 的 schedule。

---

## 3. C++ 侧 reorder (`ConvInt8TiledExecutor.cpp`)

```
loader -> signed int8 (W2 ∈ [-2,1], W3 ∈ [-4,3], W4 ∈ [-8,7])
       -> tmpWeight = signed + (-originOffset)   // unsigned uint8
       -> reorderWeight 走标准 int8 排布得到 (cellCount, UNIT*SRC_UNIT)
       -> 按 cell layout 压成最终 weightReordered:
           W4: 1 byte / 2 weight
           W2: 1 byte / 4 weight (单 byte 4 路 OC)
           W3: main 段 (同 W2 layout) + aux 段 (位平面)
```

要点:
* `_computeReorderQuantInfo` 接受 `weightBits`,内部计算 `originOffset`,直接把 zp 加偏移项写入 alpha/zero buffer。
* `weightBytes` 决定 KV 排布步长:W2=0.25,W3 sdot=`(SRC_UNIT*3+7)/8 / SRC_UNIT`(sdot=0.5,smmla=0.375),W4=0.5,W8=1。
* W2/W3 的 reorder 加了 fused fast path,避免 W4 之前用过的 2× oc·ic staging,RSS spike 显著下降(W2: 2.71 → 1.56 GB,W3: 2.60 → 1.40 GB,见提交 `feature/support_2bit`)。

---

## 4. 函数指针注册与选择

```
struct MatmulRelatedFunctions / CoreInt8Functions:
    Int8GemmKernel                        // W8 FP32
    Int8GemmKernel_W4 / _W3 / _W2         // FP32
    MNNGemmInt8AddBiasScale_Unit_FP16     // W8 FP16
    MNNGemmInt8AddBiasScale_w{4,3,2}_Unit_FP16
    MNNGemmInt8AddBiasScale_*_DecodeMax   // SME2 单 batch 优化(目前仅 W8/W4)
```

`DenseConvInt8TiledExecutor::onResize` 按 `mWeightBits + gcore->bytes/pack + 函数指针是否非空` 选 kernel。SME2 DecodeMax 暂未提供 W2/W3 版本,fallback 到普通 W2/W3 kernel;函数指针为空的 backend(老 ARMv8 / x86)按 `canUseInt2/3/4` 不走低 bit 路径,直接 loader 展开成 int8。

---

## 5. 性能数据(LinearRoofline 4096×14336, M-Mac 4t,GB/s)

| 位宽 | i8mm (smmla) | 备注 |
|------|--------------|------|
| W8   | 109.4        | 超过 llama.cpp Q8_0 (109) |
| W4   | 100.7        | +29% vs llama.cpp Q4_0 (78) |
| W3   | 50.2         | 与 llama.cpp Q3_K 持平,unsigned-domain + 4-IDX TILE_1 后从 41.3 提升 |
| W2   | 64.5         | +58% vs llama.cpp Q2_K (41) |

W2/W3 sdot path 的 prefill 大尺寸场景因 TILE_4/8/12 fall-through 偏慢,decode 已经达预期。

---

## 6. 已知限制 & 后续

| 项 | 状态 |
|----|------|
| W2/W3 sdot TILE_4/8/12 | 暂 fall-through 到 TILE_1,prefill 偏慢,需要按 W4 模板补 schedule。 |
| W2/W3 SME2 DecodeMax | 未提供专用 kernel,直接复用普通 W2/W3。 |
| W3 sdot 解包 ALU 链 | 含 ext 串行,可参考 i8mm 的 4-IDX 多寄存器思路降一档延迟。 |
| ARMv8.0(无 sdot/i8mm) | 不进入低 bit 路径,loader 展开成 int8 走 W8 kernel。 |

---

## 7. 关键文件索引

```
source/backend/cpu/arm/arm64/low_memory/                        # FP32 路径
  MNNGemmInt8AddBiasScale_ARMV82_w{2,3,4}_Unit.S                 # sdot
  MNNGemmInt8AddBiasScale_ARMV86_w{2,3,4}_Unit.S                 # i8mm
  MNNGemmInt8AddBiasScale_16x4_w4_Unit.S                         # 旧 ARMv8 路径(W4)
source/backend/cpu/arm/arm64/                                   # FP32 W8
  MNNGemmInt8AddBiasScale_ARMV82_Unit.S
  MNNGemmInt8AddBiasScale_ARMV86_Unit.S
  MNNGemmInt8AddBiasScale_16x4_Unit{,_FAST}.S                    # 老 dotprod 路径
source/backend/arm82/asm/arm64/low_memory/                      # FP16 W2/W3/W4
  MNNGemmInt8AddBiasScale_ARMV82_w{2,3,4}_Unit_FP16.S
  MNNGemmInt8AddBiasScale_ARMV86_w{2,3,4}_Unit_FP16.S
source/backend/arm82/asm/arm64/                                 # FP16 W8
  MNNGemmInt8AddBiasScale_ARMV82_Unit_FP16.S
  MNNGemmInt8AddBiasScale_ARMV86_Unit_FP16.S

source/backend/cpu/compute/
  ConvInt8TiledExecutor.cpp     # reorder / originOffset / kernel 选择
  CommonOptFunction.h           # MatmulRelatedFunctions
  Int8FunctionsOpt.{h,cpp}      # CoreInt8Functions
test/speed/GemvBWTest.cpp       # LinearRoofline benchmark (W2/W3/W4/W8)
skills/arm-cpu-optimize/        # 优化方法论与 step 文档
```

//
//  MNNSamplerC4BilinearOpt.S
//  MNN
//
//  Created by MNN on 2018/11/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5
//void MNNSamplerC4BilinearOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t iw, size_t ih, size_t yStride);
asm_function MNNSamplerC4BilinearOpt

//Auto: x0:source, x1:dest, x2:points, x3:count
//x4: xMax, x5: yMax, x6:yStride

movi v19.4s, #0

ld1 {v0.2s, v1.2s}, [x2]
//L4:
//cmp x3, #4
//blt L1
//dup v16.4s, w4
//dup v17.4s, w5
//movi v3.2s, #4
//scvtf v3.2s, v3.2s 
//fmul v3.2s, v3.2s, v1.2s
//dup v25.4s, v3.s[0]
//dup v26.4s, v3.s[1]
//
//fadd v2.2s, v0.2s, v1.2s
//mov v4.s[0], v0.s[0]
//fadd v3.2s, v2.2s, v1.2s
//mov v5.s[0], v0.s[1]
//mov v4.s[1], v2.s[0]
//mov v5.s[1], v2.s[1]
//mov v4.s[2], v3.s[0]
//fadd v2.2s, v3.2s, v1.2s
//mov v5.s[2], v3.s[1]
//mov v4.s[3], v2.s[0]
//mov v5.s[3], v2.s[1]
//
//dup v23.4s, w6
//movi v24.4s, #4
//dup v22.2d, x0
//
//L4Loop:
//fcvtns v6.4s, v4.4s
//fcvtns v7.4s, v5.4s
//
//smin v6.4s, v6.4s, v16.4s
//smin v7.4s, v7.4s, v17.4s
//smax v6.4s, v6.4s, v19.4s
//smax v7.4s, v7.4s, v19.4s
//
//mul v7.4s, v7.4s, v23.4s
//mla v7.4s, v6.4s, v24.4s
//uxtl v6.2d, v7.2s
//uxtl2 v7.2d, v7.4s
//add v6.2d, v6.2d, v22.2d
//add v7.2d, v7.2d, v22.2d
//
//mov x12, v6.d[0]
//mov x13, v6.d[1]
//ld1 {v3.s}[0], [x12]
//mov x12, v7.d[0]
//ld1 {v3.s}[1], [x13]
//fadd v5.4s, v26.4s, v5.4s
//mov x13, v7.d[1]
//ld1 {v3.s}[2], [x12]
//fadd v4.4s, v25.4s, v4.4s
//ld1 {v3.s}[3], [x13]
//
//st1 {v3.4s}, [x1], #16
//
//
//sub x3, x3, #4
//cmp x3, #4
//bge L4Loop
//
//mov v0.s[0], v4.s[0]
//mov v0.s[1], v5.s[0]


L1:
cmp x3, #0
beq End
mov v16.s[0], w4
mov v16.s[1], w5 // v16:[xMax, yMax]
mov w12, #4
mov v7.s[0], w12 // bpp=4
mov v7.s[1], w6 // yStride
dup v20.2d, x0

L1Loop:

fcvtzs v2.2s, v0.2s // [x0, y0]
frintm v4.2s, v0.2s
smax v2.2s, v2.2s, v19.2s // max(0, y)
fcvtps v3.2s, v0.2s // [x1, y1]
fabd v4.2s, v0.2s, v4.2s // (xF, yF)
smax v3.2s, v3.2s, v19.2s
smin v2.2s, v2.2s, v16.2s
smin v3.2s, v3.2s, v16.2s
mul v2.2s, v2.2s, v7.2s // [bpp * x0, y0 * yStride]
mul v3.2s, v3.2s, v7.2s // [bpp * x1, y1 * yStride]
mov v2.s[2], v3.s[0] // v2: [bpp*x0, y0*yStride, bpp*x1, y0*yStride]
mov v3.s[2], v2.s[0] // v3: [bpp*x1, y1*yStride, bpp*x0, y1*yStride]
mov v2.s[3], v2.s[1]
mov v3.s[3], v3.s[1]

uaddlp v2.2d, v2.4s // [c00, c01]
uaddlp v3.2d, v3.4s // [c11, c10]

add v2.2d, v20.2d, v2.2d
add v3.2d, v20.2d, v3.2d
mov x4, v2.d[0]
mov x5, v2.d[1]
ld1 {v5.s}[0], [x4]
ld1 {v5.s}[1], [x5]
mov x4, v3.d[0]
uxtl v5.8h, v5.8b
mov x5, v3.d[1]
ld1 {v6.s}[0], [x4]
ld1 {v6.s}[1], [x5]
uxtl v6.8h, v6.8b
//Now v2, v3 is of no use

//v2: LT, v3: RT, v5: LB, v6:BT
uxtl v2.4s, v5.4h // c00
uxtl2 v3.4s, v5.8h // c01

ucvtf v2.4s, v2.4s
uxtl v5.4s, v6.4h // c11
ucvtf v3.4s, v3.4s
uxtl2 v6.4s, v6.8h // c10
ucvtf v5.4s, v5.4s
ucvtf v6.4s, v6.4s

fsub v3.4s, v3.4s, v2.4s
fsub v5.4s, v5.4s, v6.4s
fmla v2.4s, v3.4s, v4.s[0] // (c01-c00)*xF+c00
fmla v6.4s, v5.4s, v4.s[0] // (c11-c10)*xF+c10

fsub v6.4s, v6.4s, v2.4s
fmla v2.4s, v6.4s, v4.s[1]

fcvtzs v2.4s, v2.4s
uqxtn v2.4h, v2.4s
uqxtn v2.8b, v2.8h

fadd v0.2s, v0.2s, v1.2s
subs x3, x3, #1
st1 {v2.s}[0], [x1], #4


bne L1Loop

End:

ret
#endif

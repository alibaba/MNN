//
//  MNNLineDepthWiseInt8AddBiasScaleUnit.S
//  MNN
//
//  Created by MNN on 2019/06/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNLineDepthWiseInt8AddBiasScaleUnit

// void MNNLineDepthWiseInt8AddBiasScaleUnit(int8_t* dst, const int8_t* src, const int8_t* weight, 
//      const int32_t* bias_z, size_t width, size_t src_w_step, size_t fw, size_t fh, 
//      size_t dilateX_step, size_t dilateY_step, const float* scale_z)

// Auto Load:
// x0: dst*, x1: src*, x2: weight*, x3: bias_z*
// x4: width, x5: src_w_step, x6: fw, x7: fh
// Load from sp
// x8: dilateX_step, x9: dilateY_step, x10: scale_z

ldr x8, [sp, #0]
ldr x9, [sp, #8]
ldr x10, [sp, #16]

ld1 {v6.4s}, [x3]
ld1 {v4.4s}, [x10]
mul x10, x6, x8
sub x9, x9, x10

L8:
cmp x4, #8
blt L4

mov x12, #8
mul x12, x5, x12

L8Loop:
    // load bias
    mov v16.16b, v6.16b
    mov v17.16b, v6.16b
    mov v18.16b, v6.16b
    mov v19.16b, v6.16b
    mov v20.16b, v6.16b
    mov v21.16b, v6.16b
    mov v22.16b, v6.16b
    mov v23.16b, v6.16b

    mov x13, x1
    mov x14, x2
    mov x10, x7
    L8LoopH:
        mov x11, x6
        L8LoopW:
            ld1 {v3.s}[0], [x2], #4
            dup v3.2s, v3.s[0]
            sxtl v3.8h, v3.8b
            ld1 {v0.s}[0], [x1], x5
            ld1 {v0.s}[1], [x1], x5
            subs x11, x11, #1
            sxtl v1.8h, v0.8b
            ld1 {v0.s}[2], [x1], x5
            smlal v16.4s, v3.4h, v1.4h
            ld1 {v0.s}[3], [x1], x5
            sxtl2 v2.8h, v0.16b
            smlal2 v17.4s, v3.8h, v1.8h
            ld1 {v0.s}[0], [x1], x5
            smlal v18.4s, v3.4h, v2.4h
            ld1 {v0.s}[1], [x1], x5
            ld1 {v0.s}[2], [x1], x5
            smlal2 v19.4s, v3.8h, v2.8h
            ld1 {v0.s}[3], [x1], x5
            sxtl v1.8h, v0.8b
            sxtl2 v2.8h, v0.16b
            smlal v20.4s, v3.4h, v1.4h
            smlal2 v21.4s, v3.8h, v1.8h
            smlal v22.4s, v3.4h, v2.4h
            smlal2 v23.4s, v3.8h, v2.8h

            sub x1, x1, x12
            add x1, x1, x8
            bne L8LoopW
        L8LoopWEnd:
        subs x10, x10, #1
        add x1, x1, x9
        bne L8LoopH
    
    sub x4, x4, #8
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    
    fmul v16.4s, v16.4s, v4.4s
    fmul v17.4s, v17.4s, v4.4s
    fmul v18.4s, v18.4s, v4.4s
    fmul v19.4s, v19.4s, v4.4s
    
    fcvtzs v16.4s, v16.4s
    fcvtzs v17.4s, v17.4s
    fcvtzs v18.4s, v18.4s
    fcvtzs v19.4s, v19.4s
    
    sqxtn v24.4h, v16.4s
    sqxtn2 v24.8h, v17.4s
    sqxtn v26.4h, v18.4s
    sqxtn2 v26.8h, v19.4s
    
    sqxtn v25.8b, v24.8h
    sqxtn v27.8b, v26.8h
    st1 {v25.d}[0], [x0], #8
    st1 {v27.d}[0], [x0], #8
    
    scvtf v20.4s, v20.4s
    scvtf v21.4s, v21.4s
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s
    fmul v20.4s, v20.4s, v4.4s
    fmul v21.4s, v21.4s, v4.4s
    fmul v22.4s, v22.4s, v4.4s
    fmul v23.4s, v23.4s, v4.4s
    fcvtzs v20.4s, v20.4s
    fcvtzs v21.4s, v21.4s
    fcvtzs v22.4s, v22.4s
    fcvtzs v23.4s, v23.4s

    sqxtn v24.4h, v20.4s
    sqxtn2 v24.8h, v21.4s
    sqxtn v26.4h, v22.4s
    sqxtn2 v26.8h, v23.4s

    sqxtn v25.8b, v24.8h
    sqxtn v27.8b, v26.8h
    mov x2, x14
    add x1, x13, x12
    cmp x4, #8
    st1 {v25.d}[0], [x0], #8
    st1 {v27.d}[0], [x0], #8
    bge L8Loop

L4:
cmp x4, #4
blt L1

mov x12, #4
mul x12, x5, x12

L4Loop:
    mov v16.16b, v6.16b
    mov v17.16b, v6.16b
    mov v18.16b, v6.16b
    mov v19.16b, v6.16b
    
    mov x13, x1
    mov x14, x2
    mov x10, x7
    L4LoopH:
        mov x11, x6
        L4LoopW:
            ld1 {v3.s}[0], [x2], #4
            dup v3.2s, v3.s[0]
            sxtl v3.8h, v3.8b
            ld1 {v0.s}[0], [x1], x5
            ld1 {v0.s}[1], [x1], x5
            subs x11, x11, #1
            sxtl v1.8h, v0.8b
            ld1 {v0.s}[2], [x1], x5
            ld1 {v0.s}[3], [x1], x5
            sxtl2 v2.8h, v0.16b
            smlal v16.4s, v3.4h, v1.4h
            smlal2 v17.4s, v3.8h, v1.8h
            smlal v18.4s, v3.4h, v2.4h
            smlal2 v19.4s, v3.8h, v2.8h

            sub x1, x1, x12
            add x1, x1, x8
            bne L4LoopW
        L4LoopWEnd:
        subs x10, x10, #1
        add x1, x1, x9
        bne L4LoopH
    sub x4, x4, #4
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s

    fmul v16.4s, v16.4s, v4.4s
    fmul v17.4s, v17.4s, v4.4s
    fmul v18.4s, v18.4s, v4.4s
    fmul v19.4s, v19.4s, v4.4s
    fcvtzs v16.4s, v16.4s
    fcvtzs v17.4s, v17.4s
    fcvtzs v18.4s, v18.4s
    fcvtzs v19.4s, v19.4s

    sqxtn v24.4h, v16.4s
    sqxtn2 v24.8h, v17.4s
    sqxtn v26.4h, v18.4s
    sqxtn2 v26.8h, v19.4s
    sqxtn v25.8b, v24.8h
    sqxtn v27.8b, v26.8h
    mov x2, x14
    add x1, x13, x12
    st1 {v25.8b}, [x0], #8
    cmp x4, #4
    st1 {v27.8b}, [x0], #8
    bge L4Loop


L1:
cmp x4, #0
beq End

L1Loop:
    mov v0.16b, v6.16b
    mov x10, x7
    mov x13, x1
    mov x14, x2
    L1LoopH:
        mov x11, x6
        L1LoopW:
            ld1 {v1.s}[0], [x1], x8
            ld1 {v2.s}[0], [x2], #4
            sxtl v1.8h, v1.8b
            sxtl v2.8h, v2.8b
            smlal v0.4s, v1.4h, v2.4h
            subs x11, x11, #1
            bne L1LoopW
        subs x10, x10, #1
        add x1, x1, x9
        bne L1LoopH
    
    subs x4, x4, #1
    
    scvtf v0.4s, v0.4s
    fmul v0.4s, v0.4s, v4.4s
    fcvtzs v1.4s, v0.4s
    sqxtn v2.4h, v1.4s
    sqxtn v3.8b, v2.8h
    mov x2, x14
    add x1, x13, x5
    st1 {v3.s}[0], [x0], #4
    bne L1Loop

End:
ret

#endif

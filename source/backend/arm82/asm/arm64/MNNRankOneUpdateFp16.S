//
//  MNNRankOneUpdateFp16.S
//  MNN
//
//  Created by MNN on 2026/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

// ─────────────────────────────────────────────────────────────────────
// void MNNRankOneUpdateFp16(float* S, const float* k, const float* delta, size_t dk, size_t dv)
// S[i,j] += k[i] * delta[j]   (all data is fp16, cast to float* by convention)
// x0:S  x1:k  x2:delta  x3:dk  x4:dv
// ─────────────────────────────────────────────────────────────────────
asm_function MNNRankOneUpdateFp16

cbz x3, .LRouFp16_End
cbz x4, .LRouFp16_End
lsl x5, x4, #1             // byte stride per row (fp16 = 2 bytes)

.LRouFp16_LoopRow:
    ld1r {v31.8h}, [x1], #2   // broadcast fp16 k[i]
    mov x6, x0
    mov x7, x2
    mov x8, x4

.LRouFp16_Loop16:
    cmp x8, #16
    blt .LRouFp16_Loop8
    ld1 {v0.8h, v1.8h}, [x6]
    ld1 {v4.8h, v5.8h}, [x7], #32
    fmla v0.8h, v4.8h, v31.8h
    fmla v1.8h, v5.8h, v31.8h
    st1 {v0.8h, v1.8h}, [x6], #32
    sub x8, x8, #16
    b .LRouFp16_Loop16

.LRouFp16_Loop8:
    cmp x8, #8
    blt .LRouFp16_Loop1
    ld1 {v0.8h}, [x6]
    ld1 {v4.8h}, [x7], #16
    fmla v0.8h, v4.8h, v31.8h
    st1 {v0.8h}, [x6], #16
    sub x8, x8, #8
    b .LRouFp16_Loop8

.LRouFp16_Loop1:
    cbz x8, .LRouFp16_RowDone
    ldr h0, [x6]
    ldr h4, [x7], #2
    fmadd h0, h4, h31, h0
    str h0, [x6], #2
    sub x8, x8, #1
    b .LRouFp16_Loop1

.LRouFp16_RowDone:
    add x0, x0, x5
    subs x3, x3, #1
    bne .LRouFp16_LoopRow

.LRouFp16_End:
ret


// ─────────────────────────────────────────────────────────────────────
// void MNNDualMatVecFp16(const float* S, const float* k, const float* q,
//                         float* out_k, float* out_q, size_t dk, size_t dv)
// Read-only dual MatVec: out_k = S^T @ k, out_q = S^T @ q  (all fp16)
//
// x0:S  x1:k  x2:q  x3:out_k  x4:out_q  x5:dk  x6:dv
// ─────────────────────────────────────────────────────────────────────
asm_function MNNDualMatVecFp16

stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

cbz x5, .LDmvFp16_End
cbz x6, .LDmvFp16_End

// Zero out_k and out_q
mov x7, x3
mov x8, x4
mov x9, x6
movi v8.8h, #0
.LDmvFp16_Zero8:
    cmp x9, #8
    blt .LDmvFp16_Zero1
    st1 {v8.8h}, [x7], #16
    st1 {v8.8h}, [x8], #16
    sub x9, x9, #8
    b .LDmvFp16_Zero8
.LDmvFp16_Zero1:
    cbz x9, .LDmvFp16_ZeroDone
    str h8, [x7], #2
    str h8, [x8], #2
    sub x9, x9, #1
    b .LDmvFp16_Zero1
.LDmvFp16_ZeroDone:

lsl x12, x6, #1            // byte stride per row (fp16)

.LDmvFp16_LoopRow:
    ld1r {v30.8h}, [x1], #2    // broadcast k[i]
    ld1r {v31.8h}, [x2], #2    // broadcast q[i]
    mov x8, x0                  // S row ptr
    mov x9, x3                  // out_k ptr
    mov x10, x4                 // out_q ptr
    mov x11, x6                 // remaining dv

.LDmvFp16_Loop16:
    cmp x11, #16
    blt .LDmvFp16_Loop8

    // Load S row (16 halfs)
    ld1 {v0.8h, v1.8h}, [x8], #32
    // Load out_k accumulators
    ld1 {v4.8h, v5.8h}, [x9]
    // Load out_q accumulators
    ld1 {v16.8h, v17.8h}, [x10]

    // out_k += S * k[i]
    fmla v4.8h, v0.8h, v30.8h
    fmla v5.8h, v1.8h, v30.8h

    // out_q += S * q[i]
    fmla v16.8h, v0.8h, v31.8h
    fmla v17.8h, v1.8h, v31.8h

    st1 {v4.8h, v5.8h}, [x9], #32
    st1 {v16.8h, v17.8h}, [x10], #32

    sub x11, x11, #16
    b .LDmvFp16_Loop16

.LDmvFp16_Loop8:
    cmp x11, #8
    blt .LDmvFp16_Loop1
    ld1 {v0.8h}, [x8], #16
    ld1 {v4.8h}, [x9]
    ld1 {v16.8h}, [x10]
    fmla v4.8h, v0.8h, v30.8h
    fmla v16.8h, v0.8h, v31.8h
    st1 {v4.8h}, [x9], #16
    st1 {v16.8h}, [x10], #16
    sub x11, x11, #8
    b .LDmvFp16_Loop8

.LDmvFp16_Loop1:
    cbz x11, .LDmvFp16_RowDone
    ldr h0, [x8], #2
    ldr h4, [x9]
    ldr h16, [x10]
    fmadd h4, h0, h30, h4
    fmadd h16, h0, h31, h16
    str h4, [x9], #2
    str h16, [x10], #2
    sub x11, x11, #1
    b .LDmvFp16_Loop1

.LDmvFp16_RowDone:
    add x0, x0, x12            // advance S to next row
    subs x5, x5, #1
    bne .LDmvFp16_LoopRow

.LDmvFp16_End:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret


// ─────────────────────────────────────────────────────────────────────
// void MNNDecayRankOneUpdateFp16(float* S, const float* k, const float* delta,
//                                 float decay, size_t dk, size_t dv)
// S[i,j] = decay * S[i,j] + k[i] * delta[j]  (all fp16)
//
// x0:S  x1:k  x2:delta  s0(v0.s[0]):decay(float)  x3:dk  x4:dv
// ─────────────────────────────────────────────────────────────────────
asm_function MNNDecayRankOneUpdateFp16

cbz x3, .LDruFp16_End
cbz x4, .LDruFp16_End

// Convert float decay (s0) to fp16 and broadcast
fcvt h29, s0
dup v29.8h, v29.h[0]

lsl x5, x4, #1             // byte stride per row (fp16)

.LDruFp16_LoopRow:
    ld1r {v31.8h}, [x1], #2    // broadcast k[i]
    mov x6, x0                  // S row ptr
    mov x7, x2                  // delta ptr
    mov x8, x4                  // remaining dv

.LDruFp16_Loop16:
    cmp x8, #16
    blt .LDruFp16_Loop8

    // Load S row and delta
    ld1 {v0.8h, v1.8h}, [x6]
    ld1 {v4.8h, v5.8h}, [x7], #32

    // S = decay * S + k[i] * delta
    fmul v0.8h, v0.8h, v29.8h
    fmul v1.8h, v1.8h, v29.8h
    fmla v0.8h, v4.8h, v31.8h
    fmla v1.8h, v5.8h, v31.8h

    st1 {v0.8h, v1.8h}, [x6], #32

    sub x8, x8, #16
    b .LDruFp16_Loop16

.LDruFp16_Loop8:
    cmp x8, #8
    blt .LDruFp16_Loop1
    ld1 {v0.8h}, [x6]
    ld1 {v4.8h}, [x7], #16
    fmul v0.8h, v0.8h, v29.8h
    fmla v0.8h, v4.8h, v31.8h
    st1 {v0.8h}, [x6], #16
    sub x8, x8, #8
    b .LDruFp16_Loop8

.LDruFp16_Loop1:
    cbz x8, .LDruFp16_RowDone
    ldr h0, [x6]
    ldr h4, [x7], #2
    fmul h0, h0, h29
    fmadd h0, h4, h31, h0
    str h0, [x6], #2
    sub x8, x8, #1
    b .LDruFp16_Loop1

.LDruFp16_RowDone:
    add x0, x0, x5
    subs x3, x3, #1
    bne .LDruFp16_LoopRow

.LDruFp16_End:
ret

#endif

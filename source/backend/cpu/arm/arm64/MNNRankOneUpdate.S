//
//  MNNRankOneUpdate.S
//  MNN
//
//  Created by MNN on 2026/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __aarch64__
#include "MNNAsmGlobal.h"

.text
.align 5

// ─────────────────────────────────────────────────────────────────────
// void MNNRankOneUpdateDefault(float* S, const float* k, const float* delta, size_t dk, size_t dv)
// S[i,j] += k[i] * delta[j]
// x0:S  x1:k  x2:delta  x3:dk  x4:dv
// ─────────────────────────────────────────────────────────────────────
asm_function MNNRankOneUpdateDefault

cbz x3, .LRou_End
cbz x4, .LRou_End
lsl x5, x4, #2             // byte stride per row

.LRou_LoopRow:
    ld1r {v31.4s}, [x1], #4
    mov x6, x0
    mov x7, x2
    mov x8, x4

.LRou_Loop16:
    cmp x8, #16
    blt .LRou_Loop4
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x6]
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], #64
    fmla v0.4s, v4.4s, v31.4s
    fmla v1.4s, v5.4s, v31.4s
    fmla v2.4s, v6.4s, v31.4s
    fmla v3.4s, v7.4s, v31.4s
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x6], #64
    sub x8, x8, #16
    b .LRou_Loop16

.LRou_Loop4:
    cmp x8, #4
    blt .LRou_Loop1
    ld1 {v0.4s}, [x6]
    ld1 {v4.4s}, [x7], #16
    fmla v0.4s, v4.4s, v31.4s
    st1 {v0.4s}, [x6], #16
    sub x8, x8, #4
    b .LRou_Loop4

.LRou_Loop1:
    cbz x8, .LRou_RowDone
    ldr s0, [x6]
    ldr s4, [x7], #4
    fmadd s0, s4, s31, s0
    str s0, [x6], #4
    sub x8, x8, #1
    b .LRou_Loop1

.LRou_RowDone:
    add x0, x0, x5
    subs x3, x3, #1
    bne .LRou_LoopRow

.LRou_End:
ret


// ─────────────────────────────────────────────────────────────────────
// void MNNDualMatVecDefault(const float* S, const float* k, const float* q,
//                           float* out_k, float* out_q, size_t dk, size_t dv)
// Read-only dual MatVec: out_k = S^T @ k, out_q = S^T @ q
//
// x0:S  x1:k  x2:q  x3:out_k  x4:out_q  x5:dk  x6:dv
// ─────────────────────────────────────────────────────────────────────
asm_function MNNDualMatVecDefault

stp d14, d15, [sp, #-64]!
stp d12, d13, [sp, #16]
stp d10, d11, [sp, #32]
stp d8,  d9,  [sp, #48]

cbz x5, .LDmv_End
cbz x6, .LDmv_End

// Zero out_k and out_q
mov x7, x3
mov x8, x4
mov x9, x6
movi v8.4s, #0
.LDmv_Zero4:
    cmp x9, #4
    blt .LDmv_Zero1
    st1 {v8.4s}, [x7], #16
    st1 {v8.4s}, [x8], #16
    sub x9, x9, #4
    b .LDmv_Zero4
.LDmv_Zero1:
    cbz x9, .LDmv_ZeroDone
    str s8, [x7], #4
    str s8, [x8], #4
    sub x9, x9, #1
    b .LDmv_Zero1
.LDmv_ZeroDone:

lsl x12, x6, #2            // byte stride per row (use x12, not x7)

.LDmv_LoopRow:
    ld1r {v30.4s}, [x1], #4    // broadcast k[i]
    ld1r {v31.4s}, [x2], #4    // broadcast q[i]
    mov x8, x0                  // S row ptr
    mov x9, x3                  // out_k ptr
    mov x10, x4                 // out_q ptr
    mov x11, x6                 // remaining dv

.LDmv_Loop16:
    cmp x11, #16
    blt .LDmv_Loop4

    // Load S row
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x8], #64
    // Load out_k accumulators
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x9]
    // Load out_q accumulators
    ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10]

    // out_k += S * k[i]
    fmla v4.4s, v0.4s, v30.4s
    fmla v5.4s, v1.4s, v30.4s
    fmla v6.4s, v2.4s, v30.4s
    fmla v7.4s, v3.4s, v30.4s

    // out_q += S * q[i]
    fmla v16.4s, v0.4s, v31.4s
    fmla v17.4s, v1.4s, v31.4s
    fmla v18.4s, v2.4s, v31.4s
    fmla v19.4s, v3.4s, v31.4s

    st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x9], #64
    st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x10], #64

    sub x11, x11, #16
    b .LDmv_Loop16

.LDmv_Loop4:
    cmp x11, #4
    blt .LDmv_Loop1
    ld1 {v0.4s}, [x8], #16
    ld1 {v4.4s}, [x9]
    ld1 {v16.4s}, [x10]
    fmla v4.4s, v0.4s, v30.4s
    fmla v16.4s, v0.4s, v31.4s
    st1 {v4.4s}, [x9], #16
    st1 {v16.4s}, [x10], #16
    sub x11, x11, #4
    b .LDmv_Loop4

.LDmv_Loop1:
    cbz x11, .LDmv_RowDone
    ldr s0, [x8], #4
    ldr s4, [x9]
    ldr s16, [x10]
    fmadd s4, s0, s30, s4
    fmadd s16, s0, s31, s16
    str s4, [x9], #4
    str s16, [x10], #4
    sub x11, x11, #1
    b .LDmv_Loop1

.LDmv_RowDone:
    add x0, x0, x12            // advance S to next row
    subs x5, x5, #1
    bne .LDmv_LoopRow

.LDmv_End:
ldp d8,  d9,  [sp, #48]
ldp d10, d11, [sp, #32]
ldp d12, d13, [sp, #16]
ldp d14, d15, [sp], #64
ret


// ─────────────────────────────────────────────────────────────────────
// void MNNDecayRankOneUpdateDefault(float* S, const float* k, const float* delta,
//                                    float decay, size_t dk, size_t dv)
// S[i,j] = decay * S[i,j] + k[i] * delta[j]
//
// x0:S  x1:k  x2:delta  s0(v0.s[0]):decay  x3:dk  x4:dv
// ─────────────────────────────────────────────────────────────────────
asm_function MNNDecayRankOneUpdateDefault

cbz x3, .LDru_End
cbz x4, .LDru_End

// v29 = broadcast decay
dup v29.4s, v0.s[0]

lsl x5, x4, #2             // byte stride per row

.LDru_LoopRow:
    ld1r {v31.4s}, [x1], #4    // broadcast k[i]
    mov x6, x0                  // S row ptr
    mov x7, x2                  // delta ptr
    mov x8, x4                  // remaining dv

.LDru_Loop16:
    cmp x8, #16
    blt .LDru_Loop4

    // Load S row and delta
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x6]
    ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x7], #64

    // S = decay * S + k[i] * delta
    fmul v0.4s, v0.4s, v29.4s
    fmul v1.4s, v1.4s, v29.4s
    fmul v2.4s, v2.4s, v29.4s
    fmul v3.4s, v3.4s, v29.4s
    fmla v0.4s, v4.4s, v31.4s
    fmla v1.4s, v5.4s, v31.4s
    fmla v2.4s, v6.4s, v31.4s
    fmla v3.4s, v7.4s, v31.4s

    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x6], #64

    sub x8, x8, #16
    b .LDru_Loop16

.LDru_Loop4:
    cmp x8, #4
    blt .LDru_Loop1
    ld1 {v0.4s}, [x6]
    ld1 {v4.4s}, [x7], #16
    fmul v0.4s, v0.4s, v29.4s
    fmla v0.4s, v4.4s, v31.4s
    st1 {v0.4s}, [x6], #16
    sub x8, x8, #4
    b .LDru_Loop4

.LDru_Loop1:
    cbz x8, .LDru_RowDone
    ldr s0, [x6]
    ldr s4, [x7], #4
    fmul s0, s0, s29
    fmadd s0, s4, s31, s0
    str s0, [x6], #4
    sub x8, x8, #1
    b .LDru_Loop1

.LDru_RowDone:
    add x0, x0, x5
    subs x3, x3, #1
    bne .LDru_LoopRow

.LDru_End:
ret

#endif

//
//  MNNSoftmax.S
//  MNN
//
//  Created by MNN on 2021/07/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"
.text
.align 5
//void MNNSoftmax(float* dest, const float* source, size_t size)
asm_function MNNSoftmax
        push    {r4, r5, r6, r7, r8, lr}
        bic     r3, r2, #3
        lsrs    r4, r2, #2
        vpush.64        {d8, d9, d10, d11, d12, d13, d14, d15}
        sub     sp, sp, #8
        beq     Loop_2
        vld1.32 {d16-d17}, [r1]
        cmp     r4, #1
        beq     Loop_3
        add     lr, r1, #16
        mov     ip, #1
Loop_4:
        vld1.32 {d18-d19}, [lr]!
        add     ip, ip, #1
        cmp     r4, ip
        vmax.f32        q8, q8, q9
        bne     Loop_4
Loop_3:
        vmov.32 ip, d16[0]
        vmov    s15, ip
        vmov.32 ip, d16[1]
        vmov    s12, ip
        vmov.32 ip, d17[0]
        vcmpe.f32       s15, s12
        vmov    s13, ip
        vmov.32 ip, d17[1]
        vmrs    APSR_nzcv, FPSCR
        vmov    s14, ip
        vmovle.f32      s15, s12
        vcmpe.f32       s13, s15
        vmrs    APSR_nzcv, FPSCR
        vmovpl.f32      s15, s13
        vcmpe.f32       s14, s15
        vmrs    APSR_nzcv, FPSCR
        vmovpl.f32      s15, s14
        cmp     r2, r3
        bls     Loop_25
Loop_24:
        add     lr, r1, r3, lsl #2
        mov     ip, r3
Loop_11:
        vldmia.32       lr!, {s14}
        add     ip, ip, #1
        vcmpe.f32       s14, s15
        vmrs    APSR_nzcv, FPSCR
        vmovpl.f32      s15, s14
        cmp     r2, ip
        bhi     Loop_11
        cmp     r4, #0
        beq     Loop_8
Loop_25:
        vmov.f32        q11, #0.0  @ v4sf
        mov     r5, r1
        vldr    d14, Loop_54
        vldr    d15, Loop_54+8
        mov     lr, r0
        vldr    d12, Loop_54+16
        vldr    d13, Loop_54+24
        mov     ip, #0
        vmov.f32        q12, #1.0e+0  @ v4sf
        vstr.32 s15, [sp, #4]
        vmov.f32        q5, #5.0e-1  @ v4sf
        vldr    d8, Loop_54+32
        vldr    d9, Loop_54+40
        vldr    d0, Loop_54+48
        vldr    d1, Loop_54+56
        vldr    d2, Loop_54+64
        vldr    d3, Loop_54+72
        vldr    d4, Loop_54+80
        vldr    d5, Loop_54+88
        vldr    d30, Loop_54+96
        vldr    d31, Loop_54+104
        vmov.i32        q14, #8388608  @ v4si
        vdup.32 q13, d7[1]
Loop_12:
        vld1.32 {d20-d21}, [r5]!
        add     ip, ip, #1
        vmov.i32        q3, #127  @ v4si
        cmp     r4, ip
        vsub.f32        q10, q10, q13
        vmax.f32        q10, q10, q15
        vmin.f32        q10, q10, q2
        vmul.f32        q9, q10, q6
        vcvt.s32.f32    q9, q9
        vcvt.f32.s32    q8, q9
        vadd.i32        q9, q9, q3
        vmul.f32        q8, q8, q7
        vmul.i32        q9, q9, q14
        vsub.f32        q10, q10, q8
        vmul.f32        q8, q1, q10
        vadd.f32        q8, q8, q0
        vmul.f32        q8, q8, q10
        vadd.f32        q8, q8, q4
        vmul.f32        q8, q8, q10
        vadd.f32        q8, q8, q5
        vmul.f32        q8, q8, q10
        vadd.f32        q8, q8, q12
        vmul.f32        q8, q8, q10
        vadd.f32        q8, q8, q12
        vmul.f32        q9, q9, q8
        vadd.f32        q11, q9, q11
        vst1.32 {d18-d19}, [lr]!
        bgt     Loop_12
        vmov.32 ip, d22[0]
        vldr.32 s15, [sp, #4]
        cmp     r2, r3
        vmov    s12, ip
        vmov.32 ip, d22[1]
        vmov    s11, ip
        vmov.32 ip, d23[0]
        vadd.f32        s12, s12, s11
        vmov    s13, ip
        vmov.32 ip, d23[1]
        vadd.f32        s12, s12, s13
        vmov    s14, ip
        vadd.f32        s12, s12, s14
        bls     Loop_52
Loop_26:
        lsl     ip, r3, #2
        add     r5, r1, r2, lsl #2
        add     lr, r1, ip
        movw    r7, #13877
        movt    r7, 179
        movw    r6, #55317
        movt    r6, 32310
        vldr.32 s11, Loop_54+112
        vldr.32 s10, Loop_54+116
        add     r1, r0, ip
        vldr.32 s9, Loop_54+120
        vmov.f32        s8, #5.0e-1
        vldr.32 s5, Loop_54+124
        vldr.32 s6, Loop_54+128
        vldr.32 s7, Loop_54+132
Loop_17:
        vldmia.32       lr!, {s14}
        vmov    s13, r6
        vsub.f32        s14, s14, s15
        vcmpe.f32       s14, s11
        vmrs    APSR_nzcv, FPSCR
        vmovle  s13, r7
        ble     Loop_14
        vcmpe.f32       s14, s10
        vmrs    APSR_nzcv, FPSCR
        bmi     Loop_53
Loop_14:
        vadd.f32        s12, s12, s13
        cmp     r5, lr
        vstmia.32       r1!, {s13}
        bne     Loop_17
        vmov.f32        s15, #1.0e+0
        cmp     r4, #0
        vdiv.f32        s14, s15, s12
        beq     Loop_20
Loop_23:
        vdup.32 q9, d7[0]
        mov     ip, r0
        mov     r1, #0
Loop_19:
        vld1.32 {d16-d17}, [ip]
        add     r1, r1, #1
        cmp     r4, r1
        vmul.f32        q8, q8, q9
        vst1.32 {d16-d17}, [ip]!
        bgt     Loop_19
        cmp     r2, r3
        bls     Loop_1
        lsl     ip, r3, #2
Loop_20:
        add     r0, r0, ip
Loop_21:
        vldr.32 s15, [r0]
        add     r3, r3, #1
        cmp     r2, r3
        vmul.f32        s15, s15, s14
        vstmia.32       r0!, {s15}
        bhi     Loop_21
Loop_1:
        add     sp, sp, #8
        vldm    sp!, {d8-d15}
        pop     {r4, r5, r6, r7, r8, pc}
Loop_2:
        vldr.32 s15, Loop_54+136
        cmp     r2, r3
        bhi     Loop_24
Loop_8:
        cmp     r2, r3
        vldrhi.32       s12, Loop_54+136
        bhi     Loop_26
        b       Loop_1
Loop_52:
        vmov.f32        s15, #1.0e+0
        cmp     r4, #0
        vdiv.f32        s14, s15, s12
        bne     Loop_23
        b       Loop_1
Loop_53:
        vdiv.f32        s4, s14, s9
        vmov.f32        s13, #1.0e+0
        vcvt.s32.f32    s4, s4
        vcvt.f32.s32    s3, s4
        vmov    r8, s4  @ int
        vmov.f32        s4, s7
        vmls.f32        s14, s3, s9
        vmov.f32        s3, s6
        add     r8, r8, #127
        lsl     r8, r8, #23
        vmla.f32        s3, s14, s5
        vmla.f32        s4, s3, s14
        vmov.f32        s3, s8
        vmla.f32        s3, s4, s14
        vmov.f32        s4, s13
        vmla.f32        s4, s3, s14
        vmla.f32        s13, s4, s14
        vmov    s14, r8
        vmul.f32        s13, s13, s14
        b       Loop_14
Loop_54:
        .word   1060205080
        .word   1060205080
        .word   1060205080
        .word   1060205080
        .word   1069066811
        .word   1069066811
        .word   1069066811
        .word   1069066811
        .word   1042983595
        .word   1042983595
        .word   1042983595
        .word   1042983595
        .word   1026206379
        .word   1026206379
        .word   1026206379
        .word   1026206379
        .word   1007192201
        .word   1007192201
        .word   1007192201
        .word   1007192201
        .word   1118699520
        .word   1118699520
        .word   1118699520
        .word   1118699520
        .word   -1028784128
        .word   -1028784128
        .word   -1028784128
        .word   -1028784128
        .word   -1028784128
        .word   1118699520
        .word   1060205080
        .word   1007192201
        .word   1026206379
        .word   1042983595
        .word   0
#endif
#endif

//
//  MNNGemmInt8AddBiasScale_ARMV82_w3_Unit.S
//  MNN
//
//  ARMv8.2 sdot path, fp32 output, 3-bit quantized weights (bit-plane 2+1 split).
//  Per-cell layout (UNIT=8 OC * SRC_UNIT=4 IC = 32 weights = 12 bytes packed,
//  padded to 16 bytes for 4-byte alignment):
//    bytes 0..7:  main plane (2-bit), same layout as w2 sdot
//    bytes 8..11: aux plane (1-bit, OC-major); byte k in [0..3] = OC k high bits.
//    bytes 12..15: zero padding (cell stride must be a multiple of SRC_UNIT=4).
//  Output is unsigned 0..7; C++ applies originOffset = -4 to re-center.
//  TILE_12/8/4 fall through to TILE_1 single-batch; mirrors FP16 sdot w3.
//

#if defined(__aarch64__)
#include "MNNAsmGlobal.h"

.text
.align 5

// Unpack a single w3 cell: 8 main bytes in v_w_main + 4 aux bytes in low half of v_w_aux.
// Outputs v_oc03 = OC 0-3 weights x 4 IC and v_oc47 = OC 4-7 weights x 4 IC, both [0, 7].
// Mirrors UNPACK_W3_SDOT from the FP16 sdot w3 kernel.
// v_w_main / v_w_aux are clobbered.
.macro UNPACK_W3_SDOT v_w_main, v_w_aux, v_oc03, v_oc47, \
                      v_idx_m, v_shifts_m, v_idx_a, v_shifts_a, v_t1, v_t2
    tbl  \v_t1\().16b,    {\v_w_main\().16b}, \v_idx_m\().16b
    ext  \v_w_main\().16b, \v_w_main\().16b, \v_w_main\().16b, #4
    tbl  \v_t2\().16b,    {\v_w_main\().16b}, \v_idx_m\().16b
    ushl \v_t1\().16b,    \v_t1\().16b, \v_shifts_m\().16b
    ushl \v_t2\().16b,    \v_t2\().16b, \v_shifts_m\().16b
    movi \v_w_main\().16b, #3
    and  \v_t1\().16b,    \v_t1\().16b, \v_w_main\().16b
    and  \v_t2\().16b,    \v_t2\().16b, \v_w_main\().16b

    tbl  \v_oc03\().16b,  {\v_w_aux\().16b}, \v_idx_a\().16b
    ext  \v_w_aux\().16b, \v_w_aux\().16b, \v_w_aux\().16b, #2
    tbl  \v_oc47\().16b,  {\v_w_aux\().16b}, \v_idx_a\().16b
    ushl \v_oc03\().16b,  \v_oc03\().16b, \v_shifts_a\().16b
    ushl \v_oc47\().16b,  \v_oc47\().16b, \v_shifts_a\().16b
    movi \v_w_aux\().16b, #4
    and  \v_oc03\().16b,  \v_oc03\().16b, \v_w_aux\().16b
    and  \v_oc47\().16b,  \v_oc47\().16b, \v_w_aux\().16b

    orr  \v_oc03\().16b,  \v_t1\().16b, \v_oc03\().16b
    orr  \v_oc47\().16b,  \v_t2\().16b, \v_oc47\().16b
.endm

.macro ADD_BIAS_FLOAT d0, d1, d2, d3, z0
    fadd \d0\().4s, \d0\().4s, \z0\().4s
    fadd \d1\().4s, \d1\().4s, \z0\().4s
    fadd \d2\().4s, \d2\().4s, \z0\().4s
    fadd \d3\().4s, \d3\().4s, \z0\().4s
.endm

.macro ADD_FLOAT d0, d1, d2, d3, s0, s1, s2, s3
    fadd \d0\().4s, \d0\().4s, \s0\().4s
    fadd \d1\().4s, \d1\().4s, \s1\().4s
    fadd \d2\().4s, \d2\().4s, \s2\().4s
    fadd \d3\().4s, \d3\().4s, \s3\().4s
.endm

.macro SET_BIAS d0, d1, d2, d3
    movi \d0\().16b, #0
    movi \d1\().16b, #0
    movi \d2\().16b, #0
    movi \d3\().16b, #0
.endm
.macro Int32ToFloat z0, z1, z2, z3
    scvtf \z0\().4s, \z0\().4s
    scvtf \z1\().4s, \z1\().4s
    scvtf \z2\().4s, \z2\().4s
    scvtf \z3\().4s, \z3\().4s
.endm
.macro MUL_SCALE s, d0, d1, d2, d3
    fmul \d0\().4s, \d0\().4s, \s\().4s
    fmul \d1\().4s, \d1\().4s, \s\().4s
    fmul \d2\().4s, \d2\().4s, \s\().4s
    fmul \d3\().4s, \d3\().4s, \s\().4s
.endm
.macro MUL_EXTRA_SCALE s, d0, d1, d2, d3
    fmul \d0\().4s, \d0\().4s, \s\().s[0]
    fmul \d1\().4s, \d1\().4s, \s\().s[1]
    fmul \d2\().4s, \d2\().4s, \s\().s[2]
    fmul \d3\().4s, \d3\().4s, \s\().s[3]
.endm
.macro FloatToInt32 z0, z1, z2, z3
    fcvtas \z0\().4s, \z0\().4s
    fcvtas \z1\().4s, \z1\().4s
    fcvtas \z2\().4s, \z2\().4s
    fcvtas \z3\().4s, \z3\().4s
.endm
.macro Int32ToInt16 s0, s1, s2, s3, d0, d1
    sqxtn \d0\().4h,  \s0\().4s
    sqxtn2 \d0\().8h, \s1\().4s
    sqxtn \d1\().4h,  \s2\().4s
    sqxtn2 \d1\().8h, \s3\().4s
.endm
.macro Int16ToInt8_ONE s0, s1, d0
    sqxtn \d0\().8b,   \s0\().8h
    sqxtn2 \d0\().16b, \s1\().8h
.endm
.macro Int16ToInt8 s0, s1, s2, s3, d0, d1
    Int16ToInt8_ONE \s0, \s1, \d0
    Int16ToInt8_ONE \s2, \s3, \d1
.endm
.macro MLA_WEIGHTZERO d0, s0, s1, idx // idx for xKernelSum
    fmla \d0\().4s, \s1\().4s, \s0\().s[\idx]
.endm
.macro ReLU_FP32 s0, s1, s2, s3, z0, z1 // z0:min z1:max
    fmin \s0\().4s, \s0\().4s, \z1\().4s
    fmin \s1\().4s, \s1\().4s, \z1\().4s
    fmin \s2\().4s, \s2\().4s, \z1\().4s
    fmin \s3\().4s, \s3\().4s, \z1\().4s
    fmax \s0\().4s, \s0\().4s, \z0\().4s
    fmax \s1\().4s, \s1\().4s, \z0\().4s
    fmax \s2\().4s, \s2\().4s, \z0\().4s
    fmax \s3\().4s, \s3\().4s, \z0\().4s
.endm

.macro REVERT_INPUT_DEQUANT_BIAS rg0, rg1, rg2, rg3
mul \rg1, \rg2, \rg3
sub \rg0, \rg0, \rg1
.endm

.macro REVERT_WEIGHT_KERNEL_SUM rg0, rg1, rg2, rg3
// y=UP_DIV(ocDiv4,(hp/pack))
add \rg1, \rg3, #1
lsr \rg1, \rg1, #1
// blockNum * y * (hp * sizeof(float))
mul \rg1, \rg2, \rg1
sub \rg0, \rg0, \rg1, LSL #5 // revert weight kernel sum
.endm

asm_function MNNGemmInt8AddBiasScale_ARMV82_w3_Unit
/* 
struct QuanPostTreatParameters {
    const float* scale;
    const float* biasFloat;
    int32_t maxValue;
    int32_t minValue;
    int32_t useInt8 = 1; // Save result as int8_t dataType; otherwise float32.
    float roundValuePos = 0.5f;
    float roundValueNeg = -0.5f;
    float* srcKernelSum;
    float* weightKernelSum;
    float* fp32minmax;
    ssize_t blockNum = 1;
    const int32_t* bias = nullptr;
    const float* inputScale = nullptr;
    const float* inputBias = nullptr;
    float* accumBuffer = nullptr;
};
*/

//void MNNGemmInt8AddBiasScale_ARMV82_w4_Unit(int8_t* dst, const int8_t* src, 
//    const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad,
// const QuanPostTreatParameters* parameters, size_t realDstCount);

//Auto: x0:dst, x1:src, x2:weight, x3:src_depth_quad, x4:dst_step
//x5:dst_depth_quad, x6: parameters, x7: realDstCount

//Load from x6: x9: bias, x8: xKernelSum, x23: fp32minmax
ldr x9, [x6, #8]

stp d14, d15, [sp, #(-16 * 10)]!
stp d12, d13, [sp, #(16 * 1)]
stp d10, d11, [sp, #(16 * 2)]
stp d8,  d9,  [sp, #(16 * 3)]
stp x21, x22, [sp, #(16 * 4)]
stp x19, x20, [sp, #(16 * 5)]
stp x23, x24, [sp, #(16 * 6)]
stp x25, x26, [sp, #(16 * 7)]
stp x27, x28, [sp, #(16 * 8)]

ldr x8, [x6, #40]  // srcKernelSum
ldr x28, [x6, #48] // weightKernelSum
ldr x24, [x6, #80]  // inputScale
ldr x27, [x6, #88]  // inputBias
ldr x10, [x6, #96]  // accumBuffer

ldr x23, [x6, #56]  // fp32minmax
ldr x26, [x6, #64]  // blockNum
lsl x22, x7, #2 // eDest * SRC_UNIT

// x25 = inputScale post-index stride per block in TILE_1 (4-byte ld1 .s[0]).
//   - no inputBias (x27 == 0): mInputBlockNum == 1, re-read same scale -> stride 0
//   - inputBias set (x27 != 0): mInputBlockNum == blockNum, advance by batch*4 (= x22)
mov x25, #0
cbz x27, _w3_load_consts_fp32
mov x25, x22

_w3_load_consts_fp32:
    // v6=main idx [0..3 x4], v7=main shifts [-6,-4,-2,0 in groups of 4]
    // v28=aux idx [0x8, 1x8], v29=aux shifts [2,1,0,-1,-2,-3,-4,-5] x2
    adr x16, .L_w3_sdot_unpack_consts_fp32
    ld1 {v6.16b, v7.16b},   [x16], #32
    ld1 {v28.16b, v29.16b}, [x16]

// MVP: TILE_12 / TILE_8 / TILE_4 fall through to single-batch TILE_1.
TILE_12:
TILE_8:
TILE_4:
    cbz x7, End
    b TILE_1_Init

TILE_1_Init:
    cbz x23, TILE_1
    ld1r {v26.4s}, [x23], #4 // f32 min
    ld1r {v27.4s}, [x23] // f32 max
TILE_1:
    // Already execute: [movi v7.16b, #15] in TILE_4
    mov x6, x0
    mov x12, x2
    mov x14, x5
    mov x20, x9
    mov x15, x8 // input kernel sum
    mov x21, x24 // input dequant scale
L8LoopDz_TILE_1:
    cmp x14, #2
    blt L4LoopDz_TILE_1
    mov x11, x1
    mov x19, #0
TILE1_BLOCKNUM:
    mov x13, x3

    movi v8.16b, #0           // accumulator OC 0-3
    movi v9.16b, #0           // accumulator OC 4-7

    cmp x22, #4
    bne L8LoopSz_TILE_1_lu1   // lu4 src load assumes batch=1
    cmp x13, #4
    blt L8LoopSz_TILE_1_lu1

    movi v30.16b, #0          // shadow accum OC 0-3 for ILP
    movi v31.16b, #0          // shadow accum OC 4-7

    L8LoopSz_TILE_1_lu4:
        // 4 cells of w3 weight: each cell = 16B (8B main + 4B aux + 4B pad).
        ld1 {v10.16b, v11.16b, v12.16b, v13.16b}, [x12], #64
        ld1 {v0.4s}, [x11], #16
        sub x13, x13, #4

        ext v22.16b, v10.16b, v10.16b, #8
        UNPACK_W3_SDOT v10, v22, v14, v18, v6, v7, v28, v29, v4, v5

        ext v23.16b, v11.16b, v11.16b, #8
        UNPACK_W3_SDOT v11, v23, v15, v19, v6, v7, v28, v29, v4, v5

        ext v24.16b, v12.16b, v12.16b, #8
        UNPACK_W3_SDOT v12, v24, v16, v20, v6, v7, v28, v29, v4, v5

        ext v25.16b, v13.16b, v13.16b, #8
        UNPACK_W3_SDOT v13, v25, v17, v21, v6, v7, v28, v29, v4, v5

        cmp x13, #4
        .inst 0x4f80e1c8 // sdot v8.4s,  v14.16b, v0.4b[0]
        .inst 0x4f80e249 // sdot v9.4s,  v18.16b, v0.4b[0]
        .inst 0x4fa0e1fe // sdot v30.4s, v15.16b, v0.4b[1]
        .inst 0x4fa0e27f // sdot v31.4s, v19.16b, v0.4b[1]
        .inst 0x4f80ea08 // sdot v8.4s,  v16.16b, v0.4b[2]
        .inst 0x4f80ea89 // sdot v9.4s,  v20.16b, v0.4b[2]
        .inst 0x4fa0ea3e // sdot v30.4s, v17.16b, v0.4b[3]
        .inst 0x4fa0eabf // sdot v31.4s, v21.16b, v0.4b[3]
        bge L8LoopSz_TILE_1_lu4

    add v8.4s, v8.4s, v30.4s
    add v9.4s, v9.4s, v31.4s
    cbz x13, L8LoopSzEnd_TILE_1

    L8LoopSz_TILE_1_lu1:
        // w3 cell stride = 16B: 8B main + 4B aux + 4B zero-pad
        ld1 {v4.8b},      [x12], #8                 // main plane
        ld1 {v5.s}[0],    [x12], #4                 // aux plane (4 bytes)
        add x12, x12, #4                             // skip 4-byte zero-pad
        ld1 {v0.s}[0],    [x11], x22                // src: 4 IC for 1 batch
        subs x13, x13, #1
        UNPACK_W3_SDOT v4, v5, v3, v12, v6, v7, v28, v29, v10, v11

        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
        .inst 0x4f80e189 // sdot v9.4s, v12.16b, v0.4b[0]
        bne L8LoopSz_TILE_1_lu1

    L8LoopSzEnd_TILE_1:

    L8Tile1Quan:
    ld1 {v0.4s, v1.4s}, [x12], #32 // scale
    ld1 {v2.s}[0], [x8], x22 // x kernel sum
    ld1 {v24.4s, v25.4s}, [x12], #32 // weight quan zeropoint
    scvtf v8.4s, v8.4s
    scvtf v9.4s, v9.4s
    fmul v8.4s, v8.4s, v0.4s
    fmul v9.4s, v9.4s, v1.4s

    ld1 {v4.s}[0], [x24], x25
    fmul v8.4s, v8.4s, v4.s[0]
    fmul v9.4s, v9.4s, v4.s[0]

    TILE1_L8_MLA_TERM:
    MLA_WEIGHTZERO v8, v2, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v9, v2, v25, 0 // tile:0, oc:4-7

    cbz x27, TILE1_ADD_DSTV
    ld1 {v2.s}[0], [x27], x25
    ld1 {v24.4s, v25.4s}, [x28], #32
    MLA_WEIGHTZERO v8, v2, v24, 0 // tile:0, oc:0-3
    MLA_WEIGHTZERO v9, v2, v25, 0 // tile:0, oc:4-7

    TILE1_ADD_DSTV:
    cbz x19, TILE1_L8_ACCUM_BUFFER
    ld1 {v10.4s, v11.4s}, [x10]
    fadd v8.4s, v8.4s, v10.4s
    fadd v9.4s, v9.4s, v11.4s

    TILE1_L8_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq TILE1_POST
    st1 {v8.4s, v9.4s}, [x10]
    b TILE1_BLOCKNUM

    TILE1_POST:
    sub x14, x14, #2
    cbz x9, TILE1_RELU
    ld1 {v10.4s, v11.4s}, [x20], #32
    fadd v8.4s, v8.4s, v10.4s
    fadd v9.4s, v9.4s, v11.4s
    TILE1_RELU:
    cbz x23, TILE1_STORE
    fmin v8.4s, v8.4s, v27.4s
    fmin v9.4s, v9.4s, v27.4s
    fmax v8.4s, v8.4s, v26.4s
    fmax v9.4s, v9.4s, v26.4s

    TILE1_STORE:
    st1 {v8.4s}, [x6], x4
    st1 {v9.4s}, [x6], x4

    L8Tile1LoopCheck:
    cbz x14, Tile1End
    mov x8, x15
    mov x24, x21 // revert input dequant scale
    cbz x27, L8LoopDz_TILE_1
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x22
    b L8LoopDz_TILE_1

L4LoopDz_TILE_1:
    mov x11, x1
    mov x19, #0
L4_TILE1_BLOCKNUM:
    mov x13, x3
    movi v8.16b, #0

    L4LoopSz_TILE_1_lu1:
        // w3 cell stride = 16B: 8B main + 4B aux + 4B zero-pad
        ld1 {v4.8b},      [x12], #8                 // main plane
        ld1 {v5.s}[0],    [x12], #4                 // aux plane
        add x12, x12, #4                             // skip 4-byte zero-pad
        ld1 {v0.s}[0],    [x11], x22                // src
        subs x13, x13, #1
        UNPACK_W3_SDOT v4, v5, v3, v12, v6, v7, v28, v29, v10, v11
        .inst 0x4f80e068 // sdot v8.4s, v3.16b, v0.4b[0]
    bne L4LoopSz_TILE_1_lu1

    L4Tile1Quan:
    ld1 {v0.4s}, [x12] // scale
    add x12, x12, #32
    ld1 {v2.s}[0], [x8], x22 // x kernel sum
    ld1 {v24.4s}, [x12] // weight quan zeropoint
    add x12, x12, #32
    scvtf v8.4s, v8.4s
    fmul v8.4s, v8.4s, v0.4s

    ld1 {v4.s}[0], [x24], x25
    fmul v8.4s, v8.4s, v4.s[0]

    L4_TILE1_L8_MLA_TERM:
    MLA_WEIGHTZERO v8, v2, v24, 0 // tile:0, oc:0-3

    cbz x27, L4_TILE1_ADD_DSTV
    ld1 {v2.s}[0], [x27], x25
    ld1 {v24.4s}, [x28]
    MLA_WEIGHTZERO v8, v2, v24, 0 // tile:0, oc:0-3
    add x28, x28, #32

    L4_TILE1_ADD_DSTV:
    cbz x19, L4_TILE1_L8_ACCUM_BUFFER
    ld1 {v10.4s}, [x10]
    fadd v8.4s, v8.4s, v10.4s

    L4_TILE1_L8_ACCUM_BUFFER:
    add x19, x19, #1
    cmp x19, x26
    beq L4_TILE1_POST
    st1 {v8.4s}, [x10]
    b L4_TILE1_BLOCKNUM

    L4_TILE1_POST:
    cbz x9, L4_TILE1_RELU
    ld1 {v10.4s}, [x20], #16
    fadd v8.4s, v8.4s, v10.4s
    L4_TILE1_RELU:
    cbz x23, L4_TILE1_STORE
    fmin v8.4s, v8.4s, v27.4s
    fmax v8.4s, v8.4s, v26.4s

    L4_TILE1_STORE:
    st1 {v8.4s}, [x6], x4

Tile1End:
    subs x7, x7, #1
    beq End

    add x0, x0, #16
    add x24, x21, #4
    add x1, x1, #4
    add x8, x15, #4
    cbz x27, TILE_1
    REVERT_INPUT_DEQUANT_BIAS x27, x19, x26, x22
    REVERT_WEIGHT_KERNEL_SUM x28, x14, x26, x5
    add x27, x27, #4
    b TILE_1

End:
ldp x27, x28, [sp, #(16 * 8)]
ldp x25, x26, [sp, #(16 * 7)]
ldp x23, x24, [sp, #(16 * 6)]
ldp x19, x20, [sp, #(16 * 5)]
ldp x21, x22, [sp, #(16 * 4)]
ldp d8,  d9,  [sp, #(16 * 3)]
ldp d10, d11, [sp, #(16 * 2)]
ldp d12, d13, [sp, #(16 * 1)]
ldp d14, d15, [sp], #(16 * 10)
ret

// ===== W3 sdot unpack constants (FP32) =====
.align 4
.L_w3_sdot_unpack_consts_fp32:
    // v6 main idx: replicate bytes 0..3 four times (same as w2 sdot).
    .byte 0, 1, 2, 3, 0, 1, 2, 3
    .byte 0, 1, 2, 3, 0, 1, 2, 3
    // v7 main shifts: -6/-4/-2/0 per OC group of 4 lanes (same as w2 sdot).
    .byte -6, -6, -6, -6, -4, -4, -4, -4
    .byte -2, -2, -2, -2,  0,  0,  0,  0
    // v28 aux idx: 0x8 then 1x8 — replicates aux byte 0 to lanes 0..7 (OC0/OC1
    // of 4 IC) and aux byte 1 to lanes 8..15 (OC2/OC3).
    .byte 0, 0, 0, 0, 0, 0, 0, 0
    .byte 1, 1, 1, 1, 1, 1, 1, 1
    // v29 aux shifts: bring bit i of the aux byte to bit 2 (= 4 after mask #4).
    // OC0 = bit7 -> shift -5, OC1 = bit6 -> -4, ..., OC3 = bit4 -> -2 (lo half).
    // Hi half (after ext rotates aux 2..3 down): OC4..7 shifts mirror OC0..3.
    .byte 2, 1, 0, -1, -2, -3, -4, -5
    .byte 2, 1, 0, -1, -2, -3, -4, -5

#endif // __aarch64__

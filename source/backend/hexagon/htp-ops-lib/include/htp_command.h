#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// DSP_OP_MATMUL_Q4A16_GEMV_I8_FUSED: one M=1 decode GEMV over the members of a FusedLinear group,
// reading each member's own DSP_OP_MATMUL_Q4A16_GEMV_I8 weight blob (no concatenated copy).
// params:  the 10 GEMV_I8 ints (K at [1], scale blocks at [8], asymmetric at [9]),
//          [10] gateUp, [11] memberCount, [12 + p] N of member p (DSP_GEMV_I8_MAX_SEGMENTS slots).
// inputs:  activation, then per member: weight blob, bias (fp16, fd -1 when absent).
// outputs: gateUp == 0: one per member. gateUp == 1 (members gate, up): one, holding up * silu(gate).
#define DSP_GEMV_I8_MAX_SEGMENTS 4

// DSP_OP_LINEAR_ATTN (OpType_LinearAttention, packed C4 layout, batch 1).
// inputs:  qkv, in1 (gate or raw a), in2 (beta or raw b), convWeight [Dc,1,K], gateConst ([coef, bias] fp32,
//          fd -1 unless gateFold); in1 / in2 / gateConst are fd -1 for short_conv.
// outputs: out, convState, rnnState (fd -1 for short_conv), convOut workspace (fd -1 for short_conv).
#define DSP_LINEAR_ATTN_GATED_DELTA   0
#define DSP_LINEAR_ATTN_SHORT_CONV    1
// gated_delta_rule workspace = seqLen * convDim fp32 conv output, then per-worker vector scratch.
#define DSP_LINEAR_ATTN_MAX_WORKERS   16
#define DSP_LINEAR_ATTN_SCRATCH_BYTES (88 * 128 + 80 * 1024)

struct DSPLinearAttnParam {
  int32_t type;
  int32_t seqLen;
  int32_t convDim;
  int32_t numKHeads;
  int32_t numVHeads;
  int32_t headKDim;
  int32_t headVDim;
  int32_t kernelSize;
  int32_t useL2Norm;
  int32_t gateFold;
};

// DSP_OP_GATED_RMS_NORM (OpType_GatedRMSNorm, packed C4 layout, inside % pack == 0):
//   out = (rmsnorm(x) * gamma + beta) * silu(z).
// x is [outside = batch * heads][inside]; z and out are [batch][heads * inside].
// inputs:  x, z, gamma (fp32, fd -1 when absent), beta (fp32, fd -1 when absent or zero). outputs: out.
struct DSPGatedRMSNormParam {
  int32_t batch;
  int32_t heads;
  int32_t inside;
  float   epsilon;
};

// hmx_matmulq4fp16 only double-buffers a weight panel from this many oy up, and the host mirrors it when it
// decides whether an int4 GEMM is worth splitting into row windows. Shared so the two cannot drift apart.
#define DSP_Q4_MIN_DOUBLE_BUFFER_NP 4

struct DSPCommandGroup {
    int* commands; // interleaved: cmdFd, cmdOffset, cmdSize (negative means dirty)
};

struct DSPQueueCommandGroupReq {
    unsigned int id;
    int profile;
    int groupFd;
    int groupOffset;
    int count;
    int syncGroupFd;
    int syncGroupOffset;
    int syncGroupSize;
    int profileFd;
    int profileOffset;
    int profileSize;
    int reserved;
};

struct DSPQueueCommandGroupRsp {
    unsigned int id;
    int status;
};

// type definition for op types
enum DSPOpType {
  DSP_OP_RESERVED_0 = 0,
  DSP_OP_POOL2D_FP16,
  DSP_OP_CONV_DEPTHWISE2D_FP16,
  DSP_OP_RASTER_BLIT,
  DSP_OP_UNARY,
  DSP_OP_BINARY_BLIT,
  DSP_OP_LOOP_BLIT,
  DSP_OP_TENSOR_CONVERT,
  DSP_OP_LAYER_NORM,
  DSP_OP_LAYER_NORM_PACKED,
  DSP_OP_WEIGHT_REORDER,
  DSP_OP_WEIGHT_REORDER_INT4,
  DSP_OP_IM2COL_CONVOLUTION_FP16,
  DSP_OP_SCALE,
  DSP_OP_ROPE,
  DSP_OP_ROPE_FUSE_LAYERNORM,
  DSP_OP_ADD_FUSE_LAYERNORM,
  DSP_OP_CONV1X1_DIRECT_FP16,
  DSP_OP_FLASH_ATTN,
  DSP_OP_BINARY_ELEMENTWISE,
  DSP_OP_GET_INFO,
  // INT4 (W4A16) matmul with quantized weights and FP16 activations
  DSP_OP_RESERVED_21,
  DSP_OP_MATMUL_Q4A16_FP16,
  DSP_OP_SHARED_GATHER,
  DSP_OP_ZERO,
  DSP_OP_CAST,
  DSP_OP_SELECT,
  DSP_OP_TOPKV2_K1_FP16,
  DSP_OP_SOFTMAX,
  DSP_OP_REDUCTION,
  DSP_OP_RELU6,
  DSP_OP_MASKED_REDUCTION,
  DSP_OP_TMAC_A16W1,
  DSP_OP_FLASH_ATTENTION_BLOCK,
  DSP_OP_MATMUL_Q4A16_BLOCK_FP16,
  DSP_OP_POST_ATTN_REDUCE_FUSE,
  DSP_OP_LSTM,
  DSP_OP_RELU,
  DSP_OP_BATCH_MATMUL,
  DSP_OP_PRELU,
  DSP_OP_CONV1X1_DIRECT_W8A16_SYM_PER_CHANNEL = 40,
  DSP_OP_MATMUL_Q4A16_GEMV_I8                 = 41,
  DSP_OP_MATMUL_W8A16_BLOCK_FP16              = 42,
  DSP_OP_VISION_ATTENTION_FP16                = 43,
  DSP_OP_VISION_FLASH_ATTENTION_FP16          = 44,
  DSP_OP_MATMUL_W8A16_GEMV_I8                 = 45,
  // 46 is reserved.
  DSP_OP_MATMUL_Q4A16_GEMV_I8_FUSED           = 47,
  DSP_OP_LINEAR_ATTN                          = 48,
  DSP_OP_GATED_RMS_NORM                       = 49,
  DSP_OP_COMMAND_GROUP                        = 99,
  DSP_OP_MAX                                  = 100
};

// Skel capability bits, reported in getInfo word 19 (bits 0 and 1 are reserved).
enum DSPCapability {
  DSP_CAP_GEMV_I8_FUSED  = 1u << 2,
  DSP_CAP_LINEAR_ATTN    = 1u << 3,
  DSP_CAP_GATED_RMS_NORM = 1u << 4,
  // DSP_OP_MATMUL_Q4A16_BLOCK_FP16 reads params[11] / [12] as the row stride / first row of a row window.
  DSP_CAP_Q4_ROW_WINDOW  = 1u << 5,
};

#ifdef __cplusplus
}  // extern "C"

// Host and skel are built separately and exchange these values over FastRPC.
static_assert(DSP_OP_MATMUL_Q4A16_GEMV_I8 == 41, "DSP_OP_MATMUL_Q4A16_GEMV_I8 must stay 41");
static_assert(DSP_OP_MATMUL_W8A16_GEMV_I8 == 45, "DSP_OP_MATMUL_W8A16_GEMV_I8 must stay 45");
static_assert(DSP_OP_MATMUL_Q4A16_GEMV_I8_FUSED == 47, "DSP_OP_MATMUL_Q4A16_GEMV_I8_FUSED must stay 47");
static_assert(DSP_OP_LINEAR_ATTN == 48, "DSP_OP_LINEAR_ATTN must stay 48");
static_assert(sizeof(DSPLinearAttnParam) == 40, "DSPLinearAttnParam ABI changed");
static_assert(DSP_OP_GATED_RMS_NORM == 49, "DSP_OP_GATED_RMS_NORM must stay 49");
static_assert(sizeof(DSPGatedRMSNormParam) == 16, "DSPGatedRMSNormParam ABI changed");
#endif

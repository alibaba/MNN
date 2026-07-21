#pragma once

#ifdef __cplusplus
extern "C" {
#endif

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
    DSP_OP_COMMAND_GROUP = 99,
    DSP_OP_MAX = 100
};

#ifdef __cplusplus
}
#endif

#ifndef DSP_OP_NAME_H
#define DSP_OP_NAME_H
#include "backend/hexagon/htp-ops-lib/include/htp_command.h"
inline const char* getDSPOpName(int type) {
    switch (type) {
        case DSP_OP_POOL2D_FP16: return "POOL2D_FP16";
        case DSP_OP_CONV_DEPTHWISE2D_FP16: return "CONV_DEPTHWISE2D_FP16";
        case DSP_OP_RASTER_BLIT: return "RASTER_BLIT";
        case DSP_OP_UNARY: return "UNARY";
        case DSP_OP_BINARY_BLIT: return "BINARY_BLIT";
        case DSP_OP_LOOP_BLIT: return "LOOP_BLIT";
        case DSP_OP_TENSOR_CONVERT: return "TENSOR_CONVERT";
        case DSP_OP_LAYER_NORM: return "LAYER_NORM";
        case DSP_OP_LAYER_NORM_PACKED: return "LAYER_NORM_PACKED";
        case DSP_OP_WEIGHT_REORDER: return "WEIGHT_REORDER";
        case DSP_OP_WEIGHT_REORDER_INT4: return "WEIGHT_REORDER_INT4";
        case DSP_OP_IM2COL_CONVOLUTION_FP16: return "IM2COL_CONVOLUTION_FP16";
        case DSP_OP_SCALE: return "SCALE";
        case DSP_OP_ROPE:
            return "ROPE";
        case DSP_OP_ROPE_FUSE_LAYERNORM:
            return "ROPE_FUSE_LAYERNORM";
        case DSP_OP_ADD_FUSE_LAYERNORM: return "ADD_FUSE_LAYERNORM";
        case DSP_OP_CONV1X1_DIRECT_FP16: return "CONV1X1_DIRECT_FP16";
        case DSP_OP_FLASH_ATTN: return "FLASH_ATTN";
        case DSP_OP_BINARY_ELEMENTWISE: return "BINARY_ELEMENTWISE";
        case DSP_OP_GET_INFO: return "GET_INFO";
        case DSP_OP_MATMUL_Q4A16_FP16: return "MATMUL_Q4A16_FP16";
        case DSP_OP_TMAC_A16W1: return "TMAC_A16W1";
        case DSP_OP_MATMUL_Q4A16_BLOCK_FP16: return "MATMUL_Q4A16_BLOCK_FP16";
        case DSP_OP_MATMUL_Q4A16_GEMV_I8:
            return "MATMUL_Q4A16_GEMV_I8";
        case DSP_OP_SHARED_GATHER: return "SHARED_GATHER";
        case DSP_OP_ZERO: return "ZERO";
        case DSP_OP_CAST: return "CAST";
        case DSP_OP_SELECT: return "SELECT";
        case DSP_OP_TOPKV2_K1_FP16: return "TOPKV2_K1_FP16";
        case DSP_OP_SOFTMAX: return "SOFTMAX";
        case DSP_OP_REDUCTION: return "REDUCTION";
        case DSP_OP_RELU6: return "RELU6";
        case DSP_OP_MASKED_REDUCTION: return "MASKED_REDUCTION";
        case DSP_OP_FLASH_ATTENTION_BLOCK: return "FLASH_ATTENTION_BLOCK";
        case DSP_OP_POST_ATTN_REDUCE_FUSE: return "POST_ATTN_REDUCE_FUSE";
        case DSP_OP_LSTM: return "LSTM";
        case DSP_OP_RELU: return "RELU";
        case DSP_OP_BATCH_MATMUL: return "BATCH_MATMUL";
        case DSP_OP_PRELU: return "PRELU";
        case DSP_OP_CONV1X1_DIRECT_W8A16_SYM_PER_CHANNEL: return "CONV1X1_DIRECT_W8A16_SYM_PER_CHANNEL";
        case DSP_OP_MATMUL_W8A16_BLOCK_FP16: return "MATMUL_W8A16_BLOCK_FP16";
        case DSP_OP_MATMUL_W8A16_GEMV_I8: return "MATMUL_W8A16_GEMV_I8";
        case DSP_OP_MATMUL_Q4A16_GEMV_I8_FUSED:
            return "MATMUL_Q4A16_GEMV_I8_FUSED";
        case DSP_OP_LINEAR_ATTN:
            return "LINEAR_ATTN";
        case DSP_OP_GATED_RMS_NORM:
            return "GATED_RMS_NORM";
        case DSP_OP_VISION_ATTENTION_FP16: return "VISION_ATTENTION_FP16";
        case DSP_OP_VISION_FLASH_ATTENTION_FP16: return "VISION_FLASH_ATTENTION_FP16";
        case DSP_OP_COMMAND_GROUP: return "COMMAND_GROUP";
        case 100: return "TENSOR_CONVERT_TYPE0_NC4HW4_TO_NCHW";
        case 101: return "TENSOR_CONVERT_TYPE1_NCHW_TO_NC4HW4";
        case 102: return "TENSOR_CONVERT_TYPE2_COPY";
        case 103: return "TENSOR_CONVERT_TYPE3_FLATTEN_CW";
        case 120: return "UNARY_ABS";
        case 121: return "UNARY_NEG";
        case 122: return "UNARY_GELU";
        case 123: return "UNARY_SIGMOID";
        case 124: return "UNARY_EXP";
        case 125: return "UNARY_LOG";
        case 126: return "UNARY_SILU";
        case 127: return "UNARY_TANH";
        case 128: return "UNARY_SQUARE";
        case 129: return "UNARY_SQRT";
        case 130: return "UNARY_RSQRT";
        case 131: return "UNARY_EXPM1";
        case 132: return "UNARY_COS";
        case 133: return "UNARY_SIN";
        case 200:
            return "MM_WEIGHT_DMA_WAIT_US";
        case 201:
            return "MM_DEQUANT_WAIT_US";
        case 202:
            return "MM_ACT_SHUFFLE_US";
        case 203:
            return "MM_HMX_US";
        case 204:
            return "MM_OUTBUF_WAIT_US";
        case 205:
            return "MM_SETUP_US";
        case 206:
            return "MM_STORE_INLINE_US";
        case 207:
            return "MM_PANEL_TAIL_WAIT_US";
        case 208:
            return "MM_FINAL_WAIT_US";
        case 209:
            return "MMW_DEQUANT_BUSY_US";
        case 210:
            return "MMW_STORE_BUSY_US";
        case 211:
            return "MMW_SHUFFLE_BUSY_US";
        case 212:
            return "MMW_DMA_SPIN_US";
        // Traffic / shape counters, not microseconds. Divide one by the matching phase above to get
        // the bandwidth that phase actually achieved.
        case 213:
            return "MM_KERNEL_TOTAL_US";
        case 214:
            return "MM_CALLS";
        case 215:
            return "MM_HMX_VTCM_READ_KB";
        case 216:
            return "MM_OUTPUT_DDR_WRITE_KB";
        case 217:
            return "MM_WEIGHT_DDR_READ_KB";
        case 218:
            return "MM_DEQUANT_VTCM_WRITE_KB";
        case 219:
            return "MM_MMAC";
        case 220:
            return "MM_ASYNC_STORE_CALLS";
        case 221:
            return "MM_VTCM_BUDGET_KB";
        case 222:
            return "MM_NP_CHUNK";
        case 223:
            return "MM_ACT_STAGE_REUSED";
        // Splits the activation staging (slot 202) into its DMA and HVX halves, and counts descriptors
        // so a stage can be told apart as bandwidth-bound or latency-bound: divide the KB by the us for
        // the achieved bandwidth, and by the descriptor count for the bytes per transfer.
        case 224:
            return "MM_ACT_DMA_US";
        case 225:
            return "MM_ACT_DDR_READ_KB";
        case 226:
            return "MM_ACT_DMA_DESCS";
        case 227:
            return "MM_WEIGHT_DMA_DESCS";
        case 228:
            return "MM_OX_LOOP_US";
        // Linear-attention chunked prefill stages, summed over workers (see linear_attention_ops.cc).
        case 229:
            return "LA_STATE_IO_US";
        case 230:
            return "LA_CONV_US";
        case 231:
            return "LA_TRANSPOSE_US";
        case 232:
            return "LA_GATES_US";
        case 233:
            return "LA_GRAM_US";
        case 234:
            return "LA_NORMS_US";
        case 235:
            return "LA_DECAY_US";
        case 236:
            return "LA_FWDSUB_US";
        case 237:
            return "LA_NORMALIZE_US";
        case 238:
            return "LA_TRI_IN_US";
        case 239:
            return "LA_ROWS_STATE_US";
        case 240:
            return "LA_TRI_OUT_US";
        case 241:
            return "LA_STATE_UPD_US";
        case 242:
            return "LA_OUT_STORE_US";
        case 243:
            return "LA_HEAD_TOTAL_US";
        default: return "UNKNOWN_DSP_OP";
    }
}
#endif

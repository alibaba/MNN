#include <HAP_mem.h>
#include <hexagon_types.h>
#include <qurt_memory.h>
#include <stddef.h>
#include <stdint.h>

#include "dsp/mmap_mgr.h"
#include "dsp/ops.h"
#include "dsp/worker_pool.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "htp_command.h"
#include "htp_ops.h"
#include "qurt.h"
#include "schema/current/Command_generated.h"

extern "C" {

extern AEEResult htp_ops_matmul_q4a16_fp16(uint8_t *output, uint8_t *activation, uint8_t *weight, uint8_t *bias,
                                           int32 m, int32 k, int32 n, int32 weight_type, int32 layout_type, int32 mp,
                                           int32 np, int32 kp, int32 scale_block_num, int32 scale_asymmetric);

extern AEEResult htp_ops_matmul_q4block_a16_fp16(uint8_t *output, uint8_t *activation, uint8_t *weight, uint8_t *bias,
                                                 int32 m, int32 k, int32 n, int32 weight_type, int32 layout_type,
                                                 int32 mp, int32 np, int32 kp, int32 scale_block_num,
                                                 int32 scale_asymmetric);

extern AEEResult htp_ops_tmac_a16w1_fp16(uint8_t *output, uint8_t *activation, uint8_t *weight, uint8_t *scale,
                                         uint8_t *bias, int32 m, int32 ic, int32 oc, int32 scale_block_num,
                                         int32 scale_asymmetric, int32 relu, int32 relu6, int32 output_bytes);

extern AEEResult htp_ops_pool2d_fp16(uint8_t *output, uint8_t *input, int32 batch, int32 ih, int32 iw, int32 oh,
                                     int32 ow, int32 c4, int32 kernelY, int32 kernelX, int32 strideY, int32 strideX,
                                     int32 padY, int32 padX, int32 padType, int32 countType, int32 poolType);

extern AEEResult htp_ops_conv_depthwise2d_fp16(uint8_t *output, uint8_t *input, uint8_t *weight, uint8_t *bias,
                                               int32 batch, int32 ih, int32 iw, int32 oh, int32 ow, int32 c4,
                                               int32 kernelY, int32 kernelX, int32 strideY, int32 strideX, int32 padY,
                                               int32 padX, int32 dilateY, int32 dilateX, int32 relu, int32 relu6);

extern AEEResult htp_ops_raster_blit(uint8_t *dst, uint8_t **src, int src_number, uint8_t *region, int32_t regionCount,
                                     int32_t bytes);

extern AEEResult htp_ops_unary(uint8_t *dst, uint8_t *src, int32 size, int32 type, int32 bytes);

extern AEEResult htp_ops_cast(uint8_t *dst, const uint8_t *src, int32_t size, int32_t castType);

extern AEEResult htp_ops_binary_blit(uint8_t *dst, const uint8_t *src0, const uint8_t *src1, uint8_t *region,
                                     int32 regionCount, int32 bytes, int32 type);

extern AEEResult htp_ops_loop_blit(uint8_t *dst, uint8_t *src0, uint8_t *src1, uint8_t *iter0, uint8_t *iter1,
                                   uint8_t *iter2, int32 cmdKind, int32 type, int32 bytes, uint8_t *param);
extern AEEResult htp_ops_batch_matmul(uint8_t *dst, uint8_t *src0, uint8_t *src1, uint8_t *iter0, uint8_t *iter1,
                                      uint8_t *iter2, int32 bytes, uint8_t *param);

extern AEEResult htp_ops_tensor_convert(uint8_t *dst, uint8_t *src, int32 batch, int32 area, int32 channel, int32 bytes,
                                        int32 type);

extern AEEResult htp_ops_layer_norm(uint8_t *dst, uint8_t *src, uint8_t *gamma, uint8_t *beta, int32 outter,
                                    int32 inner, float epsilon, int32 rmsNorm);

extern AEEResult htp_ops_layer_norm_packed(uint8_t *dst, uint8_t *src, uint8_t *gamma, uint8_t *beta, int32 batch,
                                           int32 channels, float epsilon, int32 rmsNorm);

extern AEEResult htp_ops_scale(uint8_t *dst, uint8_t *src, uint8_t *scaleBias, int32_t plane, int32_t cPack,
                               int32_t hasBias);

extern AEEResult htp_ops_rope(uint8_t *qOut, uint8_t *qIn, uint8_t *kOut, uint8_t *kIn, uint8_t *cEven, uint8_t *cOdd,
                              uint8_t *sEven, uint8_t *sOdd, int32 batch_seq, int32 numHead, int32 kvnumHead,
                              int32 headDim, int32 ropeDim, int32 inputC4);

extern AEEResult htp_ops_rope_fuse_layernorm(uint8_t *qOut, uint8_t *qIn, uint8_t *kOut, uint8_t *kIn, uint8_t *cEven,
                                             uint8_t *cOdd, uint8_t *sEven, uint8_t *sOdd, uint8_t *qGamma,
                                             uint8_t *kGamma, int32 batch_seq, int32 numHead, int32 kvnumHead,
                                             int32 headDim, int32 ropeDim, float qEps, float kEps, int32 qRmsNorm,
                                             int32 kRmsNorm, int32 inputC4);

extern AEEResult htp_ops_add_fuse_layernorm(uint8_t *dst, uint8_t *addOut, uint8_t *src0, uint8_t *src1, uint8_t *gamma,
                                            uint8_t *beta, int32 batch, int32 channels, float epsilon, int32 rmsNorm);

extern AEEResult htp_ops_flash_attn(uint8_t *o, uint8_t *q, uint8_t *k, uint8_t *v, uint8_t *mask, uint8_t *workspace,
                                    uint8_t *pastK, uint8_t *pastV, int32 qo_len, int32 seq_current, int32 seq_add,
                                    int32 n_heads, int32 n_kv_heads, int32 head_dim, float scale, int32 mask_stride,
                                    int32 max_kv_len, int32 value_c4);
extern AEEResult htp_ops_flash_attn_pages(uint8_t *o, uint8_t *q, uint8_t *k, uint8_t *v, uint8_t *mask,
                                          uint8_t *workspace, uint8_t **pastKPages, uint8_t **pastVPages, int32 qo_len,
                                          int32 seq_current, int32 seq_add, int32 n_heads, int32 n_kv_heads,
                                          int32 head_dim, float scale, int32 mask_stride, int32 max_kv_len,
                                          int32 page_count, int32 page_size, int32 value_c4);
extern AEEResult htp_ops_flash_attention_block(uint8_t *m_new, uint8_t *l_new, uint8_t *out_new, uint8_t *q, uint8_t *k,
                                               uint8_t *v, uint8_t *mask, uint8_t *running_m, uint8_t *running_l,
                                               uint8_t *running_out, uint8_t *workspace, int32 batch, int32 heads,
                                               int32 tokens, int32 chunk, int32 head_dim, float scale);

extern AEEResult htp_ops_getInfo_impl(uint8_t *dst);

extern AEEResult htp_ops_weight_reorder(uint8_t *dst, uint8_t *src, const WeightReorderParam *params);
extern AEEResult htp_ops_weight_reorder_int4(uint8_t *dst, uint8_t *srcInt4, int32 ic, int32 oc, int32 alphaSize);

extern AEEResult htp_ops_binary_elementwise(uint8_t *dst, uint8_t *src0, uint8_t *src1, int32 outSize, int32 in0Size,
                                            int32 in1Size, int32 type, int32 bytes, int32 inputBytes,
                                            int32 inputIsFloat, int32 outputIsFloat, const int32_t *broadcastParams,
                                            int32 broadcastParamCount);
extern AEEResult htp_ops_select(uint8_t *dst, uint8_t *cond, uint8_t *src1, uint8_t *src2, int32 outSize,
                                int32 condSize, int32 in1Size, int32 in2Size, int32 bytes, int32 condBytes,
                                int32 channelSize, int32 innerSize);
extern AEEResult htp_ops_shared_gather(uint8_t *dst, uint8_t *indices, uint8_t *weight, int32 selectSize, int32 ic,
                                       int32 oc, int32 bytes, int32 isInt4);
extern AEEResult htp_ops_zero(uint8_t *dst, int32 size);
extern AEEResult htp_ops_topkv2_k1_fp16(uint8_t *values, uint8_t *indices, uint8_t *input, int32 rowSize, int32 rows);
extern AEEResult htp_ops_softmax(uint8_t *dst, const uint8_t *src, int32 outside, int32 channel, int32 inside,
                                 int32 bytes);
extern AEEResult htp_ops_reduction(uint8_t *dst, const uint8_t *src, int32 outside, int32 reduce, int32 inside,
                                   int32 type, int32 bytes);
extern AEEResult htp_ops_masked_reduction(uint8_t *dst, const uint8_t *src, const uint8_t *mask, int32 outside,
                                          int32 reduce, int32 inside, int32 type, int32 bytes);
extern AEEResult htp_ops_relu6(uint8_t *dst, const uint8_t *src, int32 size, int32 bytes, float minValue,
                               float maxValue);
extern AEEResult htp_ops_relu(uint8_t *dst, const uint8_t *src, int32 size, int32 bytes, float slope);
extern AEEResult htp_ops_prelu(uint8_t *dst, const uint8_t *src, const uint8_t *slope, int32 size, int32 bytes,
                               int32 plane, int32 channel, int32 slopeCount, int32 pack);
extern AEEResult htp_ops_lstm(uint8_t *y, uint8_t *yh, uint8_t *yc, const uint8_t *x, const uint8_t *w,
                              const uint8_t *r, const uint8_t *b, const uint8_t *h0, const uint8_t *c0,
                              const uint8_t *packedW, const uint8_t *packedR, int32 seqLength, int32 batch,
                              int32 inputSize, int32 hiddenSize, int32 direction, int32 bytes, int32 outputCount,
                              int32 scratchBytes, uint8_t *scratch, const int32 *sizes, int32 packedWeightBytes);
extern AEEResult htp_ops_im2col_convolution_fp16(uint8_t *output, uint8_t *input, uint8_t *weight, uint8_t *bias,
                                                 const HmxIm2ColConvParam *params);

using TensorVector = flatbuffers::Vector<flatbuffers::Offset<DSPCOMMAND::Tensor>>;

static int htp_profile_unary_bucket(const int32_t *intParams) {
  if (intParams == nullptr) {
    return -1;
  }
  const int32_t opType = intParams[1];
  if (opType >= 1 && opType <= 14) {
    return 119 + opType;
  }
  return -1;
}

static void sync_tensor_vector(MmapManager *mmap_manager, const TensorVector *tensors, qurt_mem_cache_op_t cache_op) {
  if (tensors == nullptr) {
    return;
  }
  for (size_t i = 0; i < tensors->size(); i++) {
    auto tensor = tensors->Get(i);
    if (tensor->fd() >= 0 && tensor->size() > 0) {
      void *ptr = mmap_manager_get_map_local(mmap_manager, tensor->fd());
      if (ptr) {
        ptr = (void *) ((uint8_t *) ptr + tensor->offset());
        qurt_mem_cache_clean((qurt_addr_t) ptr, tensor->size(), cache_op, QURT_MEM_DCACHE);
      }
    }
  }
}

static void sync_group_tensors(MmapManager *mmap_manager, int32 syncGroupFd, int32 syncGroupOffset, int32 syncGroupSize,
                               bool before_execute) {
  if (syncGroupFd < 0 || syncGroupSize <= 0) {
    return;
  }
  void *sync_base = mmap_manager_get_map_local(mmap_manager, syncGroupFd);
  if (!sync_base) {
    return;
  }
  uint8_t *sync_ptr = (uint8_t *) sync_base + syncGroupOffset;
  if (before_execute) {
    qurt_mem_cache_clean((qurt_addr_t) sync_ptr, syncGroupSize, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
  }
  const DSPCOMMAND::SyncGroup *syncGroup = flatbuffers::GetRoot<DSPCOMMAND::SyncGroup>(sync_ptr);
  if (syncGroup) {
    sync_tensor_vector(mmap_manager, before_execute ? syncGroup->inputs() : syncGroup->outputs(),
                       before_execute ? QURT_MEM_CACHE_INVALIDATE : QURT_MEM_CACHE_FLUSH);
  }
}

int htp_execute_command(MmapManager *mmap_manager, const DSPCOMMAND::Command *command) {
  int32_t opType  = command->type();
  auto    inputs  = command->inputs();
  auto    outputs = command->outputs();
  auto    params  = command->params();

  uint8_t *mapped_ptrs[64];
  int      mapped_ptrs_size = 0;
  if (inputs->size() + outputs->size() > 64) {
    FARF(ERROR, "htp_execute_command: too many tensors inputs=%d outputs=%d", (int) inputs->size(),
         (int) outputs->size());
    return AEE_EBADPARM;
  }

  for (size_t i = 0; i < inputs->size(); i++) {
    auto tensor = inputs->Get(i);
    if (tensor->fd() >= 0) {
      void *map_ptr = mmap_manager_get_map_local(mmap_manager, tensor->fd());
      if (map_ptr) {
        mapped_ptrs[mapped_ptrs_size++] = (uint8_t *) map_ptr + tensor->offset();
      } else {
        FARF(ERROR, "htp_execute_command: mmap failed for input fd %d", tensor->fd());
        return 73;
      }
    } else {
      mapped_ptrs[mapped_ptrs_size++] = NULL;
    }
  }

  for (size_t i = 0; i < outputs->size(); i++) {
    auto tensor = outputs->Get(i);
    if (tensor->fd() >= 0) {
      void *map_ptr = mmap_manager_get_map_local(mmap_manager, tensor->fd());
      if (map_ptr) {
        mapped_ptrs[mapped_ptrs_size++] = (uint8_t *) map_ptr + tensor->offset();
      } else {
        FARF(ERROR, "htp_execute_command: mmap failed for output fd %d", tensor->fd());
        return 73;
      }
    } else {
      mapped_ptrs[mapped_ptrs_size++] = NULL;
    }
  }

  int            ret         = 0;
  const int32_t *intParams   = params ? params->data() : nullptr;
  const float   *floatParams = (const float *) intParams;

  switch (opType) {
    case DSP_OP_MATMUL_Q4A16_FP16:
      {
        int m                = intParams[0];
        int k                = intParams[1];
        int n                = intParams[2];
        int weight_type      = intParams[3];
        int layout_type      = intParams[4];
        int mp               = intParams[5];
        int np               = intParams[6];
        int kp               = intParams[7];
        int scale_block_num  = params && params->size() > 8 ? intParams[8] : 1;
        int scale_asymmetric = params && params->size() > 9 ? intParams[9] : 0;
        ret =
          htp_ops_matmul_q4a16_fp16(mapped_ptrs[inputs->size()],  // output
                                    mapped_ptrs[0],               // activation (FP16)
                                    mapped_ptrs[1],               // qweight (INT4 super-blocks)
                                    mapped_ptrs[2],               // bias (FP16)
                                    m, k, n, weight_type, layout_type, mp, np, kp, scale_block_num, scale_asymmetric);
        break;
      }
    case DSP_OP_MATMUL_Q4A16_BLOCK_FP16:
      {
        int m                = intParams[0];
        int k                = intParams[1];
        int n                = intParams[2];
        int weight_type      = intParams[3];
        int layout_type      = intParams[4];
        int mp               = intParams[5];
        int np               = intParams[6];
        int kp               = intParams[7];
        int scale_block_num  = params && params->size() > 8 ? intParams[8] : 1;
        int scale_asymmetric = params && params->size() > 9 ? intParams[9] : 0;
        ret = htp_ops_matmul_q4block_a16_fp16(mapped_ptrs[inputs->size()],  // output
                                              mapped_ptrs[0],               // activation (FP16)
                                              mapped_ptrs[1],               // qweight (INT4 super-blocks)
                                              mapped_ptrs[2],               // bias (FP16)
                                              m, k, n, weight_type, layout_type, mp, np, kp, scale_block_num,
                                              scale_asymmetric);
        break;
      }
    case DSP_OP_TMAC_A16W1:
      {
        int m                = intParams[0];
        int ic               = intParams[1];
        int oc               = intParams[2];
        int scale_block_num  = intParams[3];
        int scale_asymmetric = intParams[4];
        int relu             = intParams[5];
        int relu6            = intParams[6];
        int output_bytes     = intParams[7];
        ret = htp_ops_tmac_a16w1_fp16(mapped_ptrs[inputs->size()],  // output
                                      mapped_ptrs[0],               // activation (FP16)
                                      mapped_ptrs[1],               // packed W1 weights
                                      mapped_ptrs[2],               // FP32 block scales
                                      mapped_ptrs[3],               // bias (FP16, optional)
                                      m, ic, oc, scale_block_num, scale_asymmetric, relu, relu6, output_bytes);
        break;
      }
    case DSP_OP_POOL2D_FP16:
      {
        int batch     = intParams[0];
        int ih        = intParams[1];
        int iw        = intParams[2];
        int oh        = intParams[3];
        int ow        = intParams[4];
        int c4        = intParams[5];
        int kernelY   = intParams[6];
        int kernelX   = intParams[7];
        int strideY   = intParams[8];
        int strideX   = intParams[9];
        int padY      = intParams[10];
        int padX      = intParams[11];
        int padType   = intParams[12];
        int countType = intParams[13];
        int poolType  = intParams[14];
        ret = htp_ops_pool2d_fp16(mapped_ptrs[inputs->size()], mapped_ptrs[0], batch, ih, iw, oh, ow, c4, kernelY,
                                  kernelX, strideY, strideX, padY, padX, padType, countType, poolType);
        break;
      }
    case DSP_OP_CONV_DEPTHWISE2D_FP16:
      {
        ret = htp_ops_conv_depthwise2d_fp16(
          mapped_ptrs[inputs->size()], mapped_ptrs[0], mapped_ptrs[1], mapped_ptrs[2], intParams[0], intParams[1],
          intParams[2], intParams[3], intParams[4], intParams[5], intParams[6], intParams[7], intParams[8],
          intParams[9], intParams[10], intParams[11], intParams[12], intParams[13], intParams[14], intParams[15]);
        break;
      }
    case DSP_OP_RASTER_BLIT:
      {
        int32_t  regionCount = intParams[0];
        int32_t  bytes       = intParams[1];
        int32_t  src_number  = intParams[2];
        uint8_t *regionPtr   = (uint8_t *) &intParams[3];

        uint8_t *src_ptrs[10] = { NULL };
        for (int i = 0; i < src_number && i < 10; i++) {
          src_ptrs[i] = mapped_ptrs[i];
        }

        ret = htp_ops_raster_blit(mapped_ptrs[src_number], src_ptrs, src_number, regionPtr, regionCount, bytes);
        break;
      }
    case DSP_OP_ZERO:
      {
        ret = htp_ops_zero(mapped_ptrs[inputs->size()], intParams[0]);
        break;
      }
    case DSP_OP_UNARY:
      {
        ret = htp_ops_unary(mapped_ptrs[inputs->size()], mapped_ptrs[0], intParams[0], intParams[1], intParams[2]);
        break;
      }
    case DSP_OP_CAST:
      {
        ret = htp_ops_cast(mapped_ptrs[inputs->size()], mapped_ptrs[0], intParams[0], intParams[1]);
        break;
      }
    case DSP_OP_BINARY_BLIT:
      {
        int32_t  regionCount = intParams[0];
        int32_t  bytes       = intParams[1];
        int32_t  type        = intParams[2];
        uint8_t *regionPtr   = (uint8_t *) &intParams[3];
        ret = htp_ops_binary_blit(mapped_ptrs[inputs->size()], mapped_ptrs[0], mapped_ptrs[1], regionPtr, regionCount,
                                  bytes, type);
        break;
      }
    case DSP_OP_LOOP_BLIT:
      {
        ret =
          htp_ops_loop_blit(mapped_ptrs[inputs->size()], mapped_ptrs[0], mapped_ptrs[1], mapped_ptrs[2], mapped_ptrs[3],
                            mapped_ptrs[4], intParams[0], intParams[1], intParams[2], (uint8_t *) &intParams[3]);
        break;
      }
    case DSP_OP_BATCH_MATMUL:
      {
        ret = htp_ops_batch_matmul(mapped_ptrs[inputs->size()], mapped_ptrs[0], mapped_ptrs[1], mapped_ptrs[2],
                                   mapped_ptrs[3], mapped_ptrs[4], intParams[0], (uint8_t *) &intParams[1]);
        break;
      }
    case DSP_OP_TENSOR_CONVERT:
      {
        ret = htp_ops_tensor_convert(mapped_ptrs[inputs->size()], mapped_ptrs[0], intParams[0], intParams[1],
                                     intParams[2], intParams[3], intParams[4]);
        break;
      }
    case DSP_OP_LAYER_NORM:
      {
        ret = htp_ops_layer_norm(mapped_ptrs[3], mapped_ptrs[0], mapped_ptrs[1], mapped_ptrs[2], intParams[0],
                                 intParams[1], floatParams[2], intParams[3]);
        break;
      }
    case DSP_OP_LAYER_NORM_PACKED:
      {
        ret = htp_ops_layer_norm_packed(mapped_ptrs[3], mapped_ptrs[0], mapped_ptrs[1], mapped_ptrs[2], intParams[0],
                                        intParams[1], floatParams[2], intParams[3]);
        break;
      }
    case DSP_OP_SCALE:
      {
        ret = htp_ops_scale(mapped_ptrs[2], mapped_ptrs[0], mapped_ptrs[1], intParams[0], intParams[1],
                            params->size() > 2 ? intParams[2] : 1);
        break;
      }
    case DSP_OP_ROPE:
      {
        const size_t ropeHalfBytes = (size_t)(intParams[4] / 2) * sizeof(uint16_t);
        uint8_t *cEven = mapped_ptrs[2];
        uint8_t *cOdd  = cEven + ropeHalfBytes;
        uint8_t *sEven = mapped_ptrs[3];
        uint8_t *sOdd  = sEven + ropeHalfBytes;
        ret = htp_ops_rope(mapped_ptrs[4], mapped_ptrs[0], mapped_ptrs[5], mapped_ptrs[1], cEven,
                           cOdd, sEven, sOdd, intParams[0], intParams[1], intParams[2],
                           intParams[3], intParams[4], params && params->size() > 5 ? intParams[5] : 0);
        break;
      }
    case DSP_OP_ROPE_FUSE_LAYERNORM:
      {
        const size_t ropeHalfBytes = (size_t)(intParams[4] / 2) * sizeof(uint16_t);
        uint8_t *cEven = mapped_ptrs[2];
        uint8_t *cOdd  = cEven + ropeHalfBytes;
        uint8_t *sEven = mapped_ptrs[3];
        uint8_t *sOdd  = sEven + ropeHalfBytes;
        ret = htp_ops_rope_fuse_layernorm(mapped_ptrs[6], mapped_ptrs[0], mapped_ptrs[7], mapped_ptrs[1],
                                          cEven, cOdd, sEven, sOdd,
                                          mapped_ptrs[4], mapped_ptrs[5], intParams[0], intParams[1], intParams[2],
                                          intParams[3], intParams[4], floatParams[5], floatParams[6], intParams[7],
                                          intParams[8], params && params->size() > 9 ? intParams[9] : 0);
        break;
      }
    case DSP_OP_ADD_FUSE_LAYERNORM:
      {
        ret = htp_ops_add_fuse_layernorm(mapped_ptrs[4], mapped_ptrs[5], mapped_ptrs[0], mapped_ptrs[1], mapped_ptrs[2],
                                         mapped_ptrs[3], intParams[0], intParams[1], floatParams[2], intParams[3]);
        break;
      }
    case DSP_OP_FLASH_ATTN:
      {
        uint8_t *qPtr    = mapped_ptrs[0];
        uint8_t *kPtr    = mapped_ptrs[1];
        uint8_t *vPtr    = mapped_ptrs[2];
        uint8_t *maskPtr = NULL;
        if (inputs->size() > 3 && inputs->Get(3)->fd() >= 0) {
          maskPtr = mapped_ptrs[3];
        }
        int32_t seq_current         = intParams[1];
        int32_t seq_add             = intParams[2];
        int32_t page_count          = intParams[9];
        int32_t page_size           = intParams[10];
        int32_t page_table_capacity = (params && params->size() > 11) ? intParams[11] : 0;
        int32_t value_c4            = (params && params->size() > 12) ? intParams[12] : 0;

        uint8_t *outPtr       = mapped_ptrs[inputs->size()];
        uint8_t *workspacePtr = mapped_ptrs[inputs->size() + 1];
        if (page_count > 0 && page_table_capacity > 0) {
          if (inputs->size() <= 4 || mapped_ptrs[4] == NULL || page_count > page_table_capacity) {
            FARF(ERROR, "DSP_OP_FLASH_ATTN page table mismatch: page_count=%d capacity=%d inputs=%d", page_count,
                 page_table_capacity, (int) inputs->size());
            ret = AEE_EBADPARM;
            break;
          }
          const int32_t         *pageTable = reinterpret_cast<const int32_t *>(mapped_ptrs[4]);
          std::vector<uint8_t *> pastKPages(page_count);
          std::vector<uint8_t *> pastVPages(page_count);
          for (int i = 0; i < page_count; ++i) {
            int32_t kFd     = pageTable[4 * i + 0];
            int32_t kOffset = pageTable[4 * i + 1];
            int32_t vFd     = pageTable[4 * i + 2];
            int32_t vOffset = pageTable[4 * i + 3];
            void   *kMap    = mmap_manager_get_map_local(mmap_manager, kFd);
            void   *vMap    = mmap_manager_get_map_local(mmap_manager, vFd);
            if (kFd < 0 || vFd < 0 || kMap == NULL || vMap == NULL) {
              FARF(ERROR, "DSP_OP_FLASH_ATTN page table map failed: page=%d kfd=%d vfd=%d", i, kFd, vFd);
              ret = AEE_EBADPARM;
              break;
            }
            pastKPages[i] = (uint8_t *) kMap + kOffset;
            pastVPages[i] = (uint8_t *) vMap + vOffset;
          }
          if (ret != 0) {
            break;
          }
          ret = htp_ops_flash_attn_pages(outPtr, qPtr, kPtr, vPtr, maskPtr, workspacePtr, pastKPages.data(),
                                         pastVPages.data(), intParams[0], seq_current, seq_add, intParams[3],
                                         intParams[4], intParams[5], floatParams[6], intParams[7], intParams[8],
                                         page_count, page_size, value_c4);
        } else if (page_count > 0 && inputs->size() >= (uint32_t) (4 + 2 * page_count)) {
          std::vector<uint8_t *> pastKPages(page_count);
          std::vector<uint8_t *> pastVPages(page_count);
          for (int i = 0; i < page_count; ++i) {
            pastKPages[i] = mapped_ptrs[4 + 2 * i];
            pastVPages[i] = mapped_ptrs[5 + 2 * i];
          }
          ret = htp_ops_flash_attn_pages(outPtr, qPtr, kPtr, vPtr, maskPtr, workspacePtr, pastKPages.data(),
                                         pastVPages.data(), intParams[0], seq_current, seq_add, intParams[3],
                                         intParams[4], intParams[5], floatParams[6], intParams[7], intParams[8],
                                         page_count, page_size, value_c4);
        } else {
          if (page_count > 0) {
            FARF(ERROR, "DSP_OP_FLASH_ATTN page inputs mismatch: page_count=%d inputs=%d", page_count,
                 (int) inputs->size());
            ret = AEE_EBADPARM;
            break;
          }
          uint8_t *pastKPtr = mapped_ptrs[4];
          uint8_t *pastVPtr = mapped_ptrs[5];
          ret = htp_ops_flash_attn(outPtr, qPtr, kPtr, vPtr, maskPtr, workspacePtr, pastKPtr, pastVPtr, intParams[0],
                                   seq_current, seq_add, intParams[3], intParams[4], intParams[5], floatParams[6],
                                   intParams[7], intParams[8], value_c4);
        }
        break;
      }
    case DSP_OP_FLASH_ATTENTION_BLOCK:
      {
        ret = htp_ops_flash_attention_block(
          mapped_ptrs[inputs->size()], mapped_ptrs[inputs->size() + 1], mapped_ptrs[inputs->size() + 2], mapped_ptrs[0],
          mapped_ptrs[1], mapped_ptrs[2], inputs->size() > 3 ? mapped_ptrs[3] : NULL, mapped_ptrs[4], mapped_ptrs[5],
          mapped_ptrs[6], outputs->size() > 3 ? mapped_ptrs[inputs->size() + 3] : NULL, intParams[0], intParams[1],
          intParams[2], intParams[3], intParams[4], floatParams[5]);
        break;
      }
    case DSP_OP_BINARY_ELEMENTWISE:
      {
        ret = htp_ops_binary_elementwise(mapped_ptrs[2], mapped_ptrs[0], mapped_ptrs[1], intParams[0], intParams[1],
                                         intParams[2], intParams[3], intParams[4], intParams[5], intParams[6],
                                         intParams[7], params && params->size() > 8 ? intParams + 8 : nullptr,
                                         params ? (int32_t) params->size() - 8 : 0);
        break;
      }
    case DSP_OP_SELECT:
      {
        ret = htp_ops_select(mapped_ptrs[3], mapped_ptrs[0], mapped_ptrs[1], mapped_ptrs[2], intParams[0], intParams[1],
                             intParams[2], intParams[3], intParams[4], intParams[5], intParams[6], intParams[7]);
        break;
      }
    case DSP_OP_TOPKV2_K1_FP16:
      {
        ret = htp_ops_topkv2_k1_fp16(mapped_ptrs[inputs->size()], mapped_ptrs[inputs->size() + 1], mapped_ptrs[0],
                                     intParams[0], intParams[1]);
        break;
      }
    case DSP_OP_SOFTMAX:
      {
        ret = htp_ops_softmax(mapped_ptrs[inputs->size()], mapped_ptrs[0], intParams[0], intParams[1], intParams[2],
                              intParams[3]);
        break;
      }
    case DSP_OP_REDUCTION:
      {
        ret = htp_ops_reduction(mapped_ptrs[inputs->size()], mapped_ptrs[0], intParams[0], intParams[1], intParams[2],
                                intParams[3], intParams[4]);
        break;
      }
    case DSP_OP_MASKED_REDUCTION:
      {
        ret = htp_ops_masked_reduction(mapped_ptrs[inputs->size()], mapped_ptrs[0], mapped_ptrs[1], intParams[0],
                                       intParams[1], intParams[2], intParams[3], intParams[4]);
        break;
      }
    case DSP_OP_RELU6:
      {
        ret = htp_ops_relu6(mapped_ptrs[inputs->size()], mapped_ptrs[0], intParams[0], intParams[1], floatParams[2],
                            floatParams[3]);
        break;
      }
    case DSP_OP_RELU:
      {
        ret = htp_ops_relu(mapped_ptrs[inputs->size()], mapped_ptrs[0], intParams[0], intParams[1], floatParams[2]);
        break;
      }
    case DSP_OP_PRELU:
      {
        ret = htp_ops_prelu(mapped_ptrs[inputs->size()], mapped_ptrs[0], mapped_ptrs[1], intParams[0], intParams[1],
                            intParams[2], intParams[3], intParams[4], intParams[5]);
        break;
      }
    case DSP_OP_LSTM:
      {
        const int      outputBase  = inputs->size();
        const int      outputCount = intParams[6];
        uint8_t       *yh          = outputCount > 1 ? mapped_ptrs[outputBase + 1] : nullptr;
        uint8_t       *yc          = outputCount > 2 ? mapped_ptrs[outputBase + 2] : nullptr;
        uint8_t       *scratch     = outputs->size() > outputCount ? mapped_ptrs[outputBase + outputCount] : nullptr;
        const uint8_t *packedW     = inputs->size() > 6 ? mapped_ptrs[6] : nullptr;
        const uint8_t *packedR     = inputs->size() > 7 ? mapped_ptrs[7] : nullptr;
        const int32    packedWeightBytes = params && params->size() > 17 ? intParams[17] : 0;
        ret = htp_ops_lstm(mapped_ptrs[outputBase], yh, yc, mapped_ptrs[0], mapped_ptrs[1], mapped_ptrs[2],
                           mapped_ptrs[3], mapped_ptrs[4], mapped_ptrs[5], packedW, packedR, intParams[0], intParams[1],
                           intParams[2], intParams[3], intParams[4], intParams[5], outputCount, intParams[16], scratch,
                           intParams + 7, packedWeightBytes);
        break;
      }
    case DSP_OP_WEIGHT_REORDER:
      {
        const WeightReorderParam *reorderParams = reinterpret_cast<const WeightReorderParam *>(intParams);
        ret                                     = htp_ops_weight_reorder(mapped_ptrs[1], mapped_ptrs[0], reorderParams);
        break;
      }
    case DSP_OP_WEIGHT_REORDER_INT4:
      {
        ret = htp_ops_weight_reorder_int4(mapped_ptrs[1], mapped_ptrs[0], intParams[0], intParams[1], intParams[2]);
        break;
      }
    case DSP_OP_IM2COL_CONVOLUTION_FP16:
      {
        const HmxIm2ColConvParam *im2colParams = reinterpret_cast<const HmxIm2ColConvParam *>(intParams);
        ret = htp_ops_im2col_convolution_fp16(mapped_ptrs[inputs->size()], mapped_ptrs[0], mapped_ptrs[1],
                                              mapped_ptrs[2], im2colParams);
        break;
      }
    case DSP_OP_CONV1X1_DIRECT_FP16:
      {
        const HmxIm2ColConvParam *im2colParams = reinterpret_cast<const HmxIm2ColConvParam *>(intParams);
        ret = htp_ops_conv1x1_direct_fp16(mapped_ptrs[inputs->size()], mapped_ptrs[0], mapped_ptrs[1], mapped_ptrs[2],
                                          im2colParams);
        break;
      }
    case DSP_OP_GET_INFO:
      {
        ret = htp_ops_getInfo_impl(mapped_ptrs[0]);
        break;
      }
    case DSP_OP_SHARED_GATHER:
      {
        ret = htp_ops_shared_gather(mapped_ptrs[inputs->size()], mapped_ptrs[0], mapped_ptrs[1], intParams[0],
                                    intParams[1], intParams[2], intParams[3], intParams[4]);
        break;
      }
    default:
      ret = opType;
      break;
  }

  return ret;
}

static int execute_single_command(MmapManager *mmap_manager, int32 cmdFd, int32 cmdOffset, int32 cmdSize, int32 dirty,
                                  int *profile = nullptr) {
  void *cmd_base = NULL;
  if ((cmd_base = mmap_manager_get_map_local(mmap_manager, cmdFd)) == NULL) {
    FARF(ERROR, "execute_single_command: mmap failed for cmdFd %d", cmdFd);
    return 71;
  }

  uint8_t *cmd_ptr = (uint8_t *) cmd_base + cmdOffset;

  if (dirty) {
    qurt_mem_cache_clean((qurt_addr_t) cmd_ptr, cmdSize, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
  }

  const DSPCOMMAND::Command *command = flatbuffers::GetRoot<DSPCOMMAND::Command>(cmd_ptr);
  if (command == nullptr) {
    FARF(ERROR, "execute_single_command: Failed to parse FlatBuffers Command");
    return 72;
  }
  unsigned long long start_time = 0;
  if (profile) {
    start_time = HAP_perf_get_time_us();
  }

  int ret = htp_execute_command(mmap_manager, command);

  if (profile) {
    unsigned long long end_time = HAP_perf_get_time_us();
    int                opType   = command->type();
    profile[opType] += (int) (end_time - start_time);
    if (opType == DSP_OP_TENSOR_CONVERT) {
      auto           params    = command->params();
      const int32_t *intParams = params ? params->data() : nullptr;
      if (intParams) {
        const int convertType = intParams[4];
        if (convertType >= 0 && convertType < 4) {
          profile[100 + convertType] += (int) (end_time - start_time);
        }
      }
    } else if (opType == DSP_OP_UNARY) {
      auto           params    = command->params();
      const int32_t *intParams = params ? params->data() : nullptr;
      const int      bucket    = htp_profile_unary_bucket(intParams);
      if (bucket >= 120 && bucket <= 133) {
        profile[bucket] += (int) (end_time - start_time);
      }
    }
  }

  return ret;
}

static constexpr int kMaxCommandSize   = 4096;
static constexpr int kCommandEntrySize = 3;

static bool command_is_dirty(const int *commands, int index) {
  return commands[index * kCommandEntrySize + 2] <= 0;
}

static int command_size(const int *commands, int index) {
  const int size = commands[index * kCommandEntrySize + 2];
  return size == 0 ? kMaxCommandSize : (size < 0 ? -size : size);
}

AEEResult htp_ops_execute_command_group(remote_handle64 handle, int32 groupFd, int32 groupOffset, int32 count,
                                        int32 syncGroupFd, int32 syncGroupOffset, int32 syncGroupSize) {
  if (handle == 0 || handle == (remote_handle64) -1) {
    return AEE_EBADSTATE;
  }
  MmapManager *mmap_manager = mmap_manager_init_local();
  if (mmap_manager == nullptr) {
    return -1;
  }

  void *group_base = NULL;
  if ((group_base = mmap_manager_get_map_local(mmap_manager, groupFd)) == NULL) {
    mmap_manager_destroy_local(mmap_manager);
    return -1;
  }
  uint8_t *group_ptr = (uint8_t *) group_base + groupOffset;

  int group_size = 8 + count * kCommandEntrySize * (int) sizeof(int);
  qurt_mem_cache_clean((qurt_addr_t) group_ptr, group_size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

  int *commands = (int *) (group_ptr + 8);

  if (syncGroupFd >= 0 && syncGroupSize > 0) {
    sync_group_tensors(mmap_manager, syncGroupFd, syncGroupOffset, syncGroupSize, true);
  }

  int ret = 0;
  for (int i = 0; i < count; i++) {
    ret = execute_single_command(mmap_manager, commands[i * kCommandEntrySize], commands[i * kCommandEntrySize + 1],
                                 command_size(commands, i), command_is_dirty(commands, i));
    if (ret != 0) {
      break;
    }
  }

  if (syncGroupFd >= 0 && syncGroupSize > 0) {
    sync_group_tensors(mmap_manager, syncGroupFd, syncGroupOffset, syncGroupSize, false);
  }

  mmap_manager_destroy_local(mmap_manager);
  return ret;
}

AEEResult htp_ops_execute_command_group_profile(remote_handle64 handle, int32 groupFd, int32 groupOffset, int32 count,
                                                int32 syncGroupFd, int32 syncGroupOffset, int32 syncGroupSize,
                                                int32 profileFd, int32 profileOffset, int32 profileSize) {
  if (handle == 0 || handle == (remote_handle64) -1) {
    return AEE_EBADSTATE;
  }
  MmapManager *mmap_manager = mmap_manager_init_local();
  if (mmap_manager == nullptr) {
    return -1;
  }

  void *group_base = NULL;
  if ((group_base = mmap_manager_get_map_local(mmap_manager, groupFd)) == NULL) {
    mmap_manager_destroy_local(mmap_manager);
    return -1;
  }
  uint8_t *group_ptr = (uint8_t *) group_base + groupOffset;

  int group_size = 8 + count * kCommandEntrySize * (int) sizeof(int);
  qurt_mem_cache_clean((qurt_addr_t) group_ptr, group_size, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

  int *commands = (int *) (group_ptr + 8);

  if (syncGroupFd >= 0 && syncGroupSize > 0) {
    sync_group_tensors(mmap_manager, syncGroupFd, syncGroupOffset, syncGroupSize, true);
  }

  int *profile = NULL;
  if (profileFd >= 0) {
    void *profile_base = mmap_manager_get_map_local(mmap_manager, profileFd);
    if (profile_base) {
      profile = (int *) ((uint8_t *) profile_base + profileOffset);
      qurt_mem_cache_clean((qurt_addr_t) profile, profileSize, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }
  }

  int ret = 0;
  for (int i = 0; i < count; i++) {
    ret = execute_single_command(mmap_manager, commands[i * kCommandEntrySize], commands[i * kCommandEntrySize + 1],
                                 command_size(commands, i), command_is_dirty(commands, i), profile);
    if (ret != 0) {
      break;
    }
  }

  if (syncGroupFd >= 0 && syncGroupSize > 0) {
    sync_group_tensors(mmap_manager, syncGroupFd, syncGroupOffset, syncGroupSize, false);
  }

  if (profile != NULL) {
    qurt_mem_cache_clean((qurt_addr_t) profile, profileSize, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
  }

  mmap_manager_destroy_local(mmap_manager);
  return ret;
}

}  // extern "C"

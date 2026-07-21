#include <AEEStdErr.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <remote.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_utils.h"
#include "dsp/dma_utils.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"
#include "htp_ops.h"

extern "C" {

#define MNN_HEXAGON_LSTM_STAGE 0

static inline float htp_ops_lstm_read(const uint8_t* ptr, int index, int bytes) {
  if (bytes == 2) {
    return (float)((const __fp16*)ptr)[index];
  }
  return ((const float*)ptr)[index];
}

static inline void htp_ops_lstm_write(uint8_t* ptr, int index, int bytes, float value) {
  if (bytes == 2) {
    ((__fp16*)ptr)[index] = (__fp16)value;
    return;
  }
  ((float*)ptr)[index] = value;
}

static inline float htp_ops_lstm_fast_exp(float x) {
  if (x > 10.0f) return 22026.46484375f;
  if (x < -10.0f) return 0.00004539993f;
  const float invLn2 = 1.44269504089f;
  const float ln2 = 0.69314718056f;
  int n = (int)(x * invLn2 + (x >= 0.0f ? 0.5f : -0.5f));
  float r = x - (float)n * ln2;
  float r2 = r * r;
  float poly = 1.0f + r + r2 * (0.5f + r * (0.16666667163f + r * 0.04166666791f));
  union {
    uint32_t i;
    float f;
  } v;
  v.i = (uint32_t)(n + 127) << 23;
  return poly * v.f;
}

static inline float htp_ops_lstm_sigmoid(float x) {
  return 1.0f / (1.0f + htp_ops_lstm_fast_exp(-x));
}

static inline float htp_ops_lstm_tanh(float x) {
  return 2.0f * htp_ops_lstm_sigmoid(2.0f * x) - 1.0f;
}

static inline HVX_Vector htp_ops_lstm_sigmoid_fp16_fast_vec(HVX_Vector v,
                                                            HVX_Vector zero_v,
                                                            HVX_Vector one_v) {
  HVX_VectorPred q_v_lt_0 = Q6_Q_vcmp_gt_VhfVhf(zero_v, v);
  HVX_Vector neg_v = Q6_Vhf_vsub_VhfVhf(zero_v, v);
  HVX_Vector ax = Q6_V_vmux_QVV(q_v_lt_0, neg_v, v);

  static const uint16_t slope_bits[32] = {
      0x33f5, 0x33b7, 0x3343, 0x32a4, 0x31eb, 0x3128, 0x3067, 0x2f62,
      0x2e1b, 0x2cfd, 0x2c0a, 0x2a7b, 0x292c, 0x281a, 0x267d, 0x251c,
      0x2404, 0x224d, 0x20ef, 0x1fb8, 0x1e08, 0x1cb6, 0x1b5a, 0x19bc,
      0x1879, 0x16f9, 0x156f, 0x143c, 0x1299, 0x1124, 0x1001, 0x0e3d};
  static const uint16_t bias_bits[32] = {
      0x3800, 0x3804, 0x3812, 0x3830, 0x385e, 0x389b, 0x38e4, 0x3933,
      0x3985, 0x39d5, 0x3a22, 0x3a68, 0x3aa7, 0x3ade, 0x3b0e, 0x3b38,
      0x3b5b, 0x3b78, 0x3b91, 0x3ba5, 0x3bb6, 0x3bc4, 0x3bcf, 0x3bd9,
      0x3be0, 0x3be6, 0x3beb, 0x3bef, 0x3bf3, 0x3bf5, 0x3bf7, 0x3bf9};
  static const uint16_t edge_bits[31] = {
      0x3400, 0x3800, 0x3a00, 0x3c00, 0x3d00, 0x3e00, 0x3f00, 0x4000,
      0x4080, 0x4100, 0x4180, 0x4200, 0x4280, 0x4300, 0x4380, 0x4400,
      0x4440, 0x4480, 0x44c0, 0x4500, 0x4540, 0x4580, 0x45c0, 0x4600,
      0x4640, 0x4680, 0x46c0, 0x4700, 0x4740, 0x4780, 0x47c0};

  HVX_Vector slope = Q6_Vh_vsplat_R(slope_bits[0]);
  HVX_Vector bias = Q6_Vh_vsplat_R(0x3800);
#pragma unroll
  for (int i = 1; i < 32; ++i) {
    HVX_VectorPred q_ge_edge = Q6_Q_vcmp_gt_VhfVhf(ax, Q6_Vh_vsplat_R(edge_bits[i - 1]));
    slope = Q6_V_vmux_QVV(q_ge_edge, Q6_Vh_vsplat_R(slope_bits[i]), slope);
    bias = Q6_V_vmux_QVV(q_ge_edge, Q6_Vh_vsplat_R(bias_bits[i]), bias);
  }
  HVX_Vector y = Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vmpy_VhfVhf(ax, slope), bias);

  HVX_VectorPred q_ge_8 = Q6_Q_vcmp_gt_VhfVhf(ax, Q6_Vh_vsplat_R(0x4800));
  y = Q6_V_vmux_QVV(q_ge_8, one_v, y);
  HVX_Vector y_neg = Q6_Vhf_vsub_VhfVhf(one_v, y);
  return Q6_V_vmux_QVV(q_v_lt_0, y_neg, y);
}

static inline HVX_Vector htp_ops_lstm_tanh_fp16_fast_vec(HVX_Vector v,
                                                         HVX_Vector zero_v,
                                                         HVX_Vector one_v,
                                                         HVX_Vector two_v) {
  HVX_Vector two_x = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v, two_v));
  HVX_Vector sig = htp_ops_lstm_sigmoid_fp16_fast_vec(two_x, zero_v, one_v);
  return Q6_Vhf_vsub_VhfVhf(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(sig, two_v)), one_v);
}

static inline HVX_Vector htp_ops_lstm_sigmoid_fp16_fast28_vec(HVX_Vector v,
                                                              HVX_Vector zero_v,
                                                              HVX_Vector one_v) {
  HVX_VectorPred q_v_lt_0 = Q6_Q_vcmp_gt_VhfVhf(zero_v, v);
  HVX_Vector neg_v = Q6_Vhf_vsub_VhfVhf(zero_v, v);
  HVX_Vector ax = Q6_V_vmux_QVV(q_v_lt_0, neg_v, v);

  static const uint16_t slope_bits[28] = {
      0x33f5, 0x33b7, 0x3343, 0x32a4, 0x31eb, 0x3128, 0x3067, 0x2f62,
      0x2e1b, 0x2cfd, 0x2c0a, 0x2a7b, 0x292c, 0x281a, 0x267d, 0x251c,
      0x2404, 0x224d, 0x20ef, 0x1fb8, 0x1e08, 0x1cb6, 0x1b5a, 0x19bc,
      0x1879, 0x16f9, 0x156f, 0x143c};
  static const uint16_t bias_bits[28] = {
      0x3800, 0x3804, 0x3812, 0x3830, 0x385e, 0x389b, 0x38e4, 0x3933,
      0x3985, 0x39d5, 0x3a22, 0x3a68, 0x3aa7, 0x3ade, 0x3b0e, 0x3b38,
      0x3b5b, 0x3b78, 0x3b91, 0x3ba5, 0x3bb6, 0x3bc4, 0x3bcf, 0x3bd9,
      0x3be0, 0x3be6, 0x3beb, 0x3bef};
  static const uint16_t edge_bits[27] = {
      0x3400, 0x3800, 0x3a00, 0x3c00, 0x3d00, 0x3e00, 0x3f00, 0x4000,
      0x4080, 0x4100, 0x4180, 0x4200, 0x4280, 0x4300, 0x4380, 0x4400,
      0x4440, 0x4480, 0x44c0, 0x4500, 0x4540, 0x4580, 0x45c0, 0x4600,
      0x4640, 0x4680, 0x46c0};

  HVX_Vector slope = Q6_Vh_vsplat_R(slope_bits[0]);
  HVX_Vector bias = Q6_Vh_vsplat_R(0x3800);
#pragma unroll
  for (int i = 1; i < 28; ++i) {
    HVX_VectorPred q_ge_edge = Q6_Q_vcmp_gt_VhfVhf(ax, Q6_Vh_vsplat_R(edge_bits[i - 1]));
    slope = Q6_V_vmux_QVV(q_ge_edge, Q6_Vh_vsplat_R(slope_bits[i]), slope);
    bias = Q6_V_vmux_QVV(q_ge_edge, Q6_Vh_vsplat_R(bias_bits[i]), bias);
  }
  HVX_Vector y = Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vmpy_VhfVhf(ax, slope), bias);

  HVX_VectorPred q_ge_8 = Q6_Q_vcmp_gt_VhfVhf(ax, Q6_Vh_vsplat_R(0x4800));
  y = Q6_V_vmux_QVV(q_ge_8, one_v, y);
  HVX_Vector y_neg = Q6_Vhf_vsub_VhfVhf(one_v, y);
  return Q6_V_vmux_QVV(q_v_lt_0, y_neg, y);
}

static inline HVX_Vector htp_ops_lstm_sigmoid_fp16_fast20_vec(HVX_Vector v,
                                                              HVX_Vector zero_v,
                                                              HVX_Vector one_v) {
  HVX_VectorPred q_v_lt_0 = Q6_Q_vcmp_gt_VhfVhf(zero_v, v);
  HVX_Vector neg_v = Q6_Vhf_vsub_VhfVhf(zero_v, v);
  HVX_Vector ax = Q6_V_vmux_QVV(q_v_lt_0, neg_v, v);

  static const uint16_t slope_bits[20] = {
      0x33f5, 0x33b7, 0x3343, 0x32a4, 0x31eb, 0x3128, 0x3067, 0x2f62,
      0x2e1b, 0x2cfd, 0x2c0a, 0x2a7b, 0x292c, 0x281a, 0x267d, 0x251c,
      0x21c8, 0x1c52, 0x1665, 0x10b7};
  static const uint16_t bias_bits[20] = {
      0x3800, 0x3804, 0x3812, 0x3830, 0x385e, 0x389b, 0x38e4, 0x3933,
      0x3985, 0x39d5, 0x3a22, 0x3a68, 0x3aa7, 0x3ade, 0x3b0e, 0x3b38,
      0x3b7f, 0x3bc7, 0x3be8, 0x3bf6};
  static const uint16_t edge_bits[19] = {
      0x3400, 0x3800, 0x3a00, 0x3c00, 0x3d00, 0x3e00, 0x3f00, 0x4000,
      0x4080, 0x4100, 0x4180, 0x4200, 0x4280, 0x4300, 0x4380, 0x4400,
      0x4500, 0x4600, 0x4700};

  HVX_Vector slope = Q6_Vh_vsplat_R(slope_bits[0]);
  HVX_Vector bias = Q6_Vh_vsplat_R(0x3800);
#pragma unroll
  for (int i = 1; i < 20; ++i) {
    HVX_VectorPred q_ge_edge = Q6_Q_vcmp_gt_VhfVhf(ax, Q6_Vh_vsplat_R(edge_bits[i - 1]));
    slope = Q6_V_vmux_QVV(q_ge_edge, Q6_Vh_vsplat_R(slope_bits[i]), slope);
    bias = Q6_V_vmux_QVV(q_ge_edge, Q6_Vh_vsplat_R(bias_bits[i]), bias);
  }
  HVX_Vector y = Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vmpy_VhfVhf(ax, slope), bias);

  HVX_VectorPred q_ge_8 = Q6_Q_vcmp_gt_VhfVhf(ax, Q6_Vh_vsplat_R(0x4800));
  y = Q6_V_vmux_QVV(q_ge_8, one_v, y);
  HVX_Vector y_neg = Q6_Vhf_vsub_VhfVhf(one_v, y);
  return Q6_V_vmux_QVV(q_v_lt_0, y_neg, y);
}

static inline HVX_Vector htp_ops_lstm_tanh_fp16_fast20_vec(HVX_Vector v,
                                                           HVX_Vector zero_v,
                                                           HVX_Vector one_v,
                                                           HVX_Vector two_v) {
  HVX_Vector two_x = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(v, two_v));
  HVX_Vector sig = htp_ops_lstm_sigmoid_fp16_fast20_vec(two_x, zero_v, one_v);
  return Q6_Vhf_vsub_VhfVhf(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(sig, two_v)), one_v);
}

static inline int htp_ops_lstm_up_div(int v, int d) {
  return (v + d - 1) / d;
}

static inline void htp_ops_lstm_align_vtcm(uint8_t** vtcmPtr, size_t alignment) {
  uintptr_t ptr = (uintptr_t)(*vtcmPtr);
  ptr = (ptr + alignment - 1) & ~(alignment - 1);
  *vtcmPtr = (uint8_t*)ptr;
}

typedef struct {
  worker_synctoken_t* sync_ctx;
  const __fp16* inputGateBase;
  const __fp16* recurrentGate;
  __fp16* hidden;
  __fp16* cell;
  __fp16* y;
  const __fp16* biasBase;
  HVX_Vector zero_v;
  HVX_Vector one_v;
  HVX_Vector two_v;
  int hiddenSize;
  int gateSize;
  int yBaseIndex;
  int inputRowBase;
  int startBatch;
  int batchCount;
  int packedGateTiles;
  int hiddenTiles;
  bool inputGateInVtcm;
  bool recurrentGateInVtcm;
  bool inputGatePacked;
  bool recurrentGatePacked;
} HtpOpsLstmGateTask;

static inline HVX_Vector htp_ops_lstm_load_packed_gate_vec(const __fp16* packedGate, int row, int gateIndex,
                                                           int packedGateTiles, int hiddenTiles, int hiddenOffset) {
  const int rowBlock = row >> 5;
  const int rowInBlock = row & 31;
  const int pair = rowInBlock >> 1;
  const int nt0 = gateIndex * hiddenTiles + (hiddenOffset >> 5);
  const __fp16* tile0 = packedGate + ((size_t)rowBlock * packedGateTiles + nt0) * 1024 + pair * 64;
  const __fp16* tile1 = packedGate + ((size_t)rowBlock * packedGateTiles + nt0 + 1) * 1024 + pair * 64;
  HVX_Vector v0 = Q6_Vh_vdeal_Vh(vmem((const HVX_Vector*)tile0));
  HVX_Vector v1 = Q6_Vh_vdeal_Vh(vmem((const HVX_Vector*)tile1));
  if ((rowInBlock & 1) != 0) {
    v0 = Q6_V_valign_VVR(v0, v0, 64);
    v1 = Q6_V_valign_VVR(v1, v1, 64);
  }
  return Q6_V_vmux_QVV(Q6_Q_vsetq_R(64), v0, Q6_V_valign_VVR(v1, v1, 64));
}

static inline void htp_ops_lstm_gate_update_range(const HtpOpsLstmGateTask* task) {
  const int hiddenSize = task->hiddenSize;
  const int gateSize = task->gateSize;
  if (task->batchCount > 1 && !task->inputGatePacked && !task->recurrentGatePacked) {
    const int gateBytes = gateSize * (int)sizeof(__fp16);
    const int stateBytes = hiddenSize * (int)sizeof(__fp16);
    if (!task->inputGateInVtcm) {
      l2fetch(task->inputGateBase + (size_t)task->startBatch * gateSize, gateBytes, gateBytes, task->batchCount, 0);
    }
    if (!task->recurrentGateInVtcm) {
      l2fetch(task->recurrentGate + (size_t)task->startBatch * gateSize, gateBytes, gateBytes, task->batchCount, 0);
    }
    l2fetch(task->cell + (size_t)task->startBatch * hiddenSize, stateBytes, stateBytes, task->batchCount, 0);
  }
  for (int h = 0; h < hiddenSize; h += 64) {
    HVX_Vector biasI = vmemu((const HVX_Vector*)(task->biasBase + h));
    HVX_Vector biasO = vmemu((const HVX_Vector*)(task->biasBase + hiddenSize + h));
    HVX_Vector biasF = vmemu((const HVX_Vector*)(task->biasBase + 2 * hiddenSize + h));
    HVX_Vector biasC = vmemu((const HVX_Vector*)(task->biasBase + 3 * hiddenSize + h));
    for (int b = 0; b < task->batchCount; ++b) {
      const int batchIndex = task->startBatch + b;
      const int inputPackedRow = task->inputRowBase + batchIndex;
      const __fp16* inputGateBase = task->inputGateBase + (size_t)inputPackedRow * gateSize;
      const __fp16* recurrentGateBase = task->recurrentGate + (size_t)batchIndex * gateSize;
      const int stateIndex = batchIndex * hiddenSize + h;
      const int yIndex = task->yBaseIndex + stateIndex;
      HVX_Vector inputI = task->inputGatePacked ? htp_ops_lstm_load_packed_gate_vec(task->inputGateBase, inputPackedRow, 0, task->packedGateTiles, task->hiddenTiles, h)
                                                : vmem((const HVX_Vector*)(inputGateBase + h));
      HVX_Vector inputO = task->inputGatePacked ? htp_ops_lstm_load_packed_gate_vec(task->inputGateBase, inputPackedRow, 1, task->packedGateTiles, task->hiddenTiles, h)
                                                : vmem((const HVX_Vector*)(inputGateBase + hiddenSize + h));
      HVX_Vector inputF = task->inputGatePacked ? htp_ops_lstm_load_packed_gate_vec(task->inputGateBase, inputPackedRow, 2, task->packedGateTiles, task->hiddenTiles, h)
                                                : vmem((const HVX_Vector*)(inputGateBase + 2 * hiddenSize + h));
      HVX_Vector inputC = task->inputGatePacked ? htp_ops_lstm_load_packed_gate_vec(task->inputGateBase, inputPackedRow, 3, task->packedGateTiles, task->hiddenTiles, h)
                                                : vmem((const HVX_Vector*)(inputGateBase + 3 * hiddenSize + h));
      HVX_Vector recurI = task->recurrentGatePacked ? htp_ops_lstm_load_packed_gate_vec(task->recurrentGate, batchIndex, 0, task->packedGateTiles, task->hiddenTiles, h)
                                                    : vmem((const HVX_Vector*)(recurrentGateBase + h));
      HVX_Vector recurO = task->recurrentGatePacked ? htp_ops_lstm_load_packed_gate_vec(task->recurrentGate, batchIndex, 1, task->packedGateTiles, task->hiddenTiles, h)
                                                    : vmem((const HVX_Vector*)(recurrentGateBase + hiddenSize + h));
      HVX_Vector recurF = task->recurrentGatePacked ? htp_ops_lstm_load_packed_gate_vec(task->recurrentGate, batchIndex, 2, task->packedGateTiles, task->hiddenTiles, h)
                                                    : vmem((const HVX_Vector*)(recurrentGateBase + 2 * hiddenSize + h));
      HVX_Vector recurC = task->recurrentGatePacked ? htp_ops_lstm_load_packed_gate_vec(task->recurrentGate, batchIndex, 3, task->packedGateTiles, task->hiddenTiles, h)
                                                    : vmem((const HVX_Vector*)(recurrentGateBase + 3 * hiddenSize + h));
      HVX_Vector gateI = Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vadd_VhfVhf(inputI, recurI), biasI);
      HVX_Vector gateO = Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vadd_VhfVhf(inputO, recurO), biasO);
      HVX_Vector gateF = Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vadd_VhfVhf(inputF, recurF), biasF);
      HVX_Vector gateC = Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vadd_VhfVhf(inputC, recurC), biasC);
      HVX_Vector inputGateValue = htp_ops_lstm_sigmoid_fp16_fast20_vec(gateI, task->zero_v, task->one_v);
      HVX_Vector outputGateValue = htp_ops_lstm_sigmoid_fp16_fast20_vec(gateO, task->zero_v, task->one_v);
      HVX_Vector forgetGateValue = htp_ops_lstm_sigmoid_fp16_fast20_vec(gateF, task->zero_v, task->one_v);
      HVX_Vector cellGate = htp_ops_lstm_tanh_fp16_fast20_vec(gateC, task->zero_v, task->one_v, task->two_v);
      HVX_Vector oldCell = vmem((const HVX_Vector*)(task->cell + stateIndex));
      HVX_Vector forgetPart = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(forgetGateValue, oldCell));
      HVX_Vector inputPart = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(inputGateValue, cellGate));
      HVX_Vector cellValue = Q6_Vhf_vadd_VhfVhf(forgetPart, inputPart);
      HVX_Vector hiddenValue = Q6_Vhf_equals_Vqf16(
          Q6_Vqf16_vmpy_VhfVhf(outputGateValue,
                               htp_ops_lstm_tanh_fp16_fast20_vec(cellValue, task->zero_v, task->one_v, task->two_v)));
      vmem((HVX_Vector*)(task->cell + stateIndex)) = cellValue;
      vmem((HVX_Vector*)(task->hidden + stateIndex)) = hiddenValue;
      vmem((HVX_Vector*)(task->y + yIndex)) = hiddenValue;
    }
  }
}

static void htp_ops_lstm_gate_worker(void* data, int worker_id) {
  (void)worker_id;
  HtpOpsLstmGateTask* task = (HtpOpsLstmGateTask*)data;
  htp_ops_lstm_gate_update_range(task);
  worker_pool_synctoken_jobdone(task->sync_ctx);
}

static inline int htp_ops_lstm_pick_gate_tasks(int batch) {
  if (batch < 32 || g_max_num_workers <= 1) {
    return 1;
  }
  int tasks = (int)g_max_num_workers;
  if (tasks > 4) {
    tasks = 4;
  }
  return tasks < 1 ? 1 : tasks;
}

static inline void htp_ops_lstm_hmx_pack_activation(__fp16* dst, const __fp16* src, int rows, int rowStride,
                                                    int rowBase, int validRows, int kTiles) {
  for (int kt = 0; kt < kTiles; kt += 2) {
    __fp16* tile0 = dst + (size_t)kt * 1024;
    __fp16* tile1 = tile0 + 1024;
    const int kBegin = kt * 32;
    int r = 0;
    for (; r <= validRows - 2; r += 2) {
      const __fp16* src0 = src + (rowBase + r) * rowStride + kBegin;
      const __fp16* src1 = src + (rowBase + r + 1) * rowStride + kBegin;
      HVX_Vector v0 = vmem((const HVX_Vector*)src0);
      HVX_Vector v1 = vmem((const HVX_Vector*)src1);
      HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, 64);
      vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
      vmem((HVX_Vector*)((uint8_t*)tile1 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));
    }
    if (r < validRows) {
      const __fp16* src0 = src + (rowBase + r) * rowStride + kBegin;
      HVX_Vector v0 = vmem((const HVX_Vector*)src0);
      HVX_Vector v1 = Q6_V_vzero();
      HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, 64);
      vmem((HVX_Vector*)((uint8_t*)tile0 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_lo_W(vp));
      vmem((HVX_Vector*)((uint8_t*)tile1 + r * 64)) = Q6_Vh_vshuff_Vh(Q6_V_hi_W(vp));
    }
  }
  (void)rows;
}

static inline void htp_ops_lstm_hmx_pack_weight_tile(__fp16* dst, const __fp16* weight, int n, int kSize,
                                                     int kStride, int nStride, int nKStride, int nt, int kt) {
  __fp16* tile = dst + ((size_t)nt * nKStride + kt) * 1024;
  memset(tile, 0, 1024 * sizeof(__fp16));
  const int kBegin = kt * 32;
  int nBegin = nt * 32;
  int nRemain = n - nBegin;
  if (nRemain > 32) {
    nRemain = 32;
  }
  for (int k = 0; k < 32; ++k) {
    const int rawK = kBegin + k;
    if (rawK >= kSize) {
      continue;
    }
    const __fp16* srcRow = weight + rawK * kStride;
    for (int c = 0; c < nRemain; ++c) {
      const int dstIndex = (k / 2) * 64 + c * 2 + (k & 1);
      tile[dstIndex] = srcRow[(nBegin + c) * nStride];
    }
  }
}

static inline void htp_ops_lstm_store_32_fp16(__fp16* dst, HVX_Vector value) {
  const HVX_VectorPred q_low = Q6_Q_vsetq_R(64);
  const uintptr_t addr = (uintptr_t)dst;
  if ((addr & 127) == 0) {
    Q6_vmem_QRIV(q_low, (HVX_Vector*)dst, value);
    return;
  }
  if ((addr & 127) == 64) {
    HVX_VectorPred q_high = Q6_Q_not_Q(q_low);
    Q6_vmem_QRIV(q_high, (HVX_Vector*)(addr - 64), Q6_V_valign_VVR(value, value, 64));
    return;
  }
  vstu_variable(dst, 32 * (uint32_t)sizeof(__fp16), value);
}

static inline void htp_ops_lstm_hmx_store(__fp16* dst, const __fp16* vtcmOutput, int rowStride, int rows,
                                           int n, int rowBase, int validRows, int nt) {
  const int nBegin = nt * 32;
  int nRemain = n - nBegin;
  if (nRemain > 32) {
    nRemain = 32;
  }
  if (nRemain == 32) {
    const HVX_Vector* src = (const HVX_Vector*)vtcmOutput;
    const HVX_VectorPred q_low = Q6_Q_vsetq_R(64);
    const uintptr_t firstAddr = (uintptr_t)(dst + (size_t)rowBase * rowStride + nBegin);
    const int storeAlign = (int)(firstAddr & 127);
    int r = 0;
    if (storeAlign == 0) {
      for (; r <= validRows - 2; r += 2) {
        HVX_Vector v = Q6_Vh_vdeal_Vh(*src++);
        __fp16* dst0 = dst + (rowBase + r) * rowStride + nBegin;
        __fp16* dst1 = dst + (rowBase + r + 1) * rowStride + nBegin;
        Q6_vmem_QRIV(q_low, (HVX_Vector*)dst0, v);
        Q6_vmem_QRIV(q_low, (HVX_Vector*)dst1, Q6_V_valign_VVR(v, v, 64));
      }
      if (r < validRows) {
        HVX_Vector v = Q6_Vh_vdeal_Vh(*src++);
        __fp16* dst0 = dst + (rowBase + r) * rowStride + nBegin;
        Q6_vmem_QRIV(q_low, (HVX_Vector*)dst0, v);
      }
    } else if (storeAlign == 64) {
      const HVX_VectorPred q_high = Q6_Q_not_Q(q_low);
      for (; r <= validRows - 2; r += 2) {
        HVX_Vector v = Q6_Vh_vdeal_Vh(*src++);
        uintptr_t dst0 = (uintptr_t)(dst + (rowBase + r) * rowStride + nBegin);
        uintptr_t dst1 = (uintptr_t)(dst + (rowBase + r + 1) * rowStride + nBegin);
        Q6_vmem_QRIV(q_high, (HVX_Vector*)(dst0 - 64), Q6_V_valign_VVR(v, v, 64));
        Q6_vmem_QRIV(q_high, (HVX_Vector*)(dst1 - 64), v);
      }
      if (r < validRows) {
        HVX_Vector v = Q6_Vh_vdeal_Vh(*src++);
        uintptr_t dst0 = (uintptr_t)(dst + (rowBase + r) * rowStride + nBegin);
        Q6_vmem_QRIV(q_high, (HVX_Vector*)(dst0 - 64), Q6_V_valign_VVR(v, v, 64));
      }
    } else {
      for (; r <= validRows - 2; r += 2) {
        HVX_Vector v = Q6_Vh_vdeal_Vh(*src++);
        __fp16* dst0 = dst + (rowBase + r) * rowStride + nBegin;
        __fp16* dst1 = dst + (rowBase + r + 1) * rowStride + nBegin;
        htp_ops_lstm_store_32_fp16(dst0, v);
        htp_ops_lstm_store_32_fp16(dst1, Q6_V_valign_VVR(v, v, 64));
      }
      if (r < validRows) {
        HVX_Vector v = Q6_Vh_vdeal_Vh(*src++);
        __fp16* dst0 = dst + (rowBase + r) * rowStride + nBegin;
        htp_ops_lstm_store_32_fp16(dst0, v);
      }
    }
    (void)rows;
    return;
  }
  for (int r = 0; r < validRows; ++r) {
    __fp16* dstRow = dst + (rowBase + r) * rowStride + nBegin;
    for (int c = 0; c < nRemain; ++c) {
      const int srcIndex = (r / 2) * 64 + c * 2 + (r & 1);
      dstRow[c] = vtcmOutput[srcIndex];
    }
  }
  (void)rows;
}

static inline void htp_ops_lstm_hmx_pack_weight_all(__fp16* vtcmWeight, const __fp16* weight, int n, int kSize,
                                                    int weightKStride, int weightNStride, int weightKTilesStride) {
  const int np = htp_ops_lstm_up_div(n, 32);
  const int kp = htp_ops_lstm_up_div(kSize, 32);
  for (int nt = 0; nt < np; ++nt) {
    for (int kt = 0; kt < kp; ++kt) {
      htp_ops_lstm_hmx_pack_weight_tile(vtcmWeight, weight, n, kSize, weightKStride, weightNStride,
                                        weightKTilesStride, nt, kt);
    }
  }
}

static inline void htp_ops_lstm_dma_copy_weight(__fp16* vtcmWeight, const __fp16* packedWeight, int bytes) {
  if (bytes <= 0) {
    return;
  }
  _Alignas(64) dma_desc_1d_t desc;
  memset(&desc, 0, sizeof(desc));
  desc.length = (uint32_t)bytes;
  desc.type = DMA_DESC_TYPE_1D;
  desc.src_bypass = 1;
  desc.dst_bypass = 1;
  desc.ordered = 1;
  desc.dstate = DMA_DESC_DSTATE_PENDING;
  desc.src = (uint32_t)packedWeight;
  desc.dst = (uint32_t)vtcmWeight;
  dma_wait_for_idle();
  dmstart(&desc);
  dma_wait_for_idle();
}

static inline bool htp_ops_lstm_hmx_matmul_prepacked_k64(__fp16* dst, const __fp16* src,
                                                          const __fp16* vtcmWeight, int rows, int n,
                                                          int kTiles, int weightKTilesStride,
                                                          int srcRowStride, int dstRowStride,
                                                          __fp16* vtcmActivation, __fp16* vtcmOutput,
                                                          __fp16* vtcmScales) {
  if (rows <= 0 || n <= 0 || n % 32 != 0) {
    return false;
  }
  const int np = htp_ops_lstm_up_div(n, 32);

  for (int rowBase = 0; rowBase < rows; rowBase += 32) {
    int validRows = rows - rowBase;
    if (validRows > 32) {
      validRows = 32;
    }
    htp_ops_lstm_hmx_pack_activation(vtcmActivation, src, rows, srcRowStride, rowBase, validRows, kTiles);
    for (int nt = 0; nt < np; ++nt) {
      hmx_load_tiles_fp16(vtcmActivation, vtcmWeight + (size_t)nt * weightKTilesStride * 1024, kTiles);
      hmx_consume_accumulator_fp16(vtcmOutput);
      htp_ops_lstm_hmx_store(dst, vtcmOutput, dstRowStride, rows, n, rowBase, validRows, nt);
    }
  }
  return true;
}

static inline bool htp_ops_lstm_hmx_matmul_prepacked_k64_packed(__fp16* packedDst, const __fp16* src,
                                                                 const __fp16* vtcmWeight, int rows, int n,
                                                                 int kTiles, int weightKTilesStride,
                                                                 int srcRowStride, __fp16* vtcmActivation) {
  if (rows <= 0 || n <= 0 || n % 32 != 0) {
    return false;
  }
  const int np = htp_ops_lstm_up_div(n, 32);

  for (int rowBase = 0; rowBase < rows; rowBase += 32) {
    int validRows = rows - rowBase;
    if (validRows > 32) {
      validRows = 32;
    }
    htp_ops_lstm_hmx_pack_activation(vtcmActivation, src, rows, srcRowStride, rowBase, validRows, kTiles);
    const int rowBlock = rowBase >> 5;
    for (int nt = 0; nt < np; ++nt) {
      hmx_load_tiles_fp16(vtcmActivation, vtcmWeight + (size_t)nt * weightKTilesStride * 1024, kTiles);
      __fp16* packedTile = packedDst + ((size_t)rowBlock * np + nt) * 1024;
      hmx_consume_accumulator_fp16(packedTile);
    }
  }
  return true;
}

static inline AEEResult htp_ops_lstm_fast_fp16(uint8_t* y, uint8_t* yh, uint8_t* yc, const uint8_t* x,
                                                const uint8_t* w, const uint8_t* r, const uint8_t* b,
                                                const uint8_t* h0, const uint8_t* c0,
                                                const uint8_t* packedW, const uint8_t* packedR,
                                                int32_t seqLength, int32_t batch, int32_t inputSize,
                                                int32_t hiddenSize, int32_t direction, int32_t outputCount,
                                                uint8_t* scratch, int32_t scratchBytes,
                                                const int32_t* sizes, int32_t packedWeightBytes) {
  if ((inputSize & 63) != 0 || (hiddenSize & 63) != 0 || direction <= 0) {
    return AEE_EUNSUPPORTED;
  }
  const int gateSize = 4 * hiddenSize;
  const int inputKp = htp_ops_lstm_up_div(inputSize, 32);
  const int recurrentKp = htp_ops_lstm_up_div(hiddenSize, 32);
  const int weightKpStride = inputKp > recurrentKp ? inputKp : recurrentKp;
  const int xSize = sizes[0];
  const int wSize = sizes[1];
  const int rSize = sizes[2];
  const int bSize = sizes[3];
  const int h0Size = sizes[4];
  const int c0Size = sizes[5];
  const int ySize = sizes[6];
  const int yhSize = sizes[7];
  const int ycSize = sizes[8];
  const int stateSize = batch * hiddenSize;
  const int biasDirectionStride = direction > 0 ? bSize / direction : gateSize;
  const size_t stateBytes = (size_t)stateSize * sizeof(__fp16);
  const size_t requiredScratchBytes = 2 * stateBytes;
  if (scratch == nullptr || scratchBytes < 0 || (size_t)scratchBytes < requiredScratchBytes) {
    return AEE_ENOMEMORY;
  }
  uint8_t* scratchPtr = scratch;
  __fp16* hidden = (__fp16*)scratchPtr;
  scratchPtr += stateBytes;
  __fp16* cell = (__fp16*)scratchPtr;
  uint8_t* vtcmPtr = (uint8_t*)vtcm_manager_get_vtcm_base();
  const int np = htp_ops_lstm_up_div(gateSize, 32);
  __fp16* vtcmActivation = (__fp16*)vtcm_seq_alloc(&vtcmPtr, (size_t)weightKpStride * 1024 * sizeof(__fp16));
  __fp16* vtcmInputWeight = (__fp16*)vtcm_seq_alloc(&vtcmPtr, (size_t)np * weightKpStride * 1024 * sizeof(__fp16));
  __fp16* vtcmRecurrentWeight = (__fp16*)vtcm_seq_alloc(&vtcmPtr, (size_t)np * weightKpStride * 1024 * sizeof(__fp16));
  __fp16* vtcmOutput = (__fp16*)vtcm_seq_alloc(&vtcmPtr, 1024 * sizeof(__fp16));
  __fp16* vtcmScales = (__fp16*)vtcm_seq_alloc(&vtcmPtr, 256);
  htp_ops_lstm_align_vtcm(&vtcmPtr, 2048);
  constexpr int inputGateBlockSteps = 16;
  const int inputGateBlockRows = inputGateBlockSteps * batch;
  const int inputGateBlockRowBlocks = htp_ops_lstm_up_div(inputGateBlockRows, 32);
  const int recurrentGateRowBlocks = htp_ops_lstm_up_div(batch, 32);
  __fp16* vtcmStepInputGate = (__fp16*)vtcm_seq_alloc(&vtcmPtr,
                                                      inputGateBlockRowBlocks * (size_t)np * 1024 * sizeof(__fp16));
  __fp16* vtcmRecurrentGate = (__fp16*)vtcm_seq_alloc(&vtcmPtr,
                                                      recurrentGateRowBlocks * (size_t)np * 1024 * sizeof(__fp16));
  if (vtcmActivation == nullptr || vtcmInputWeight == nullptr || vtcmRecurrentWeight == nullptr ||
      vtcmOutput == nullptr || vtcmScales == nullptr || vtcmStepInputGate == nullptr || vtcmRecurrentGate == nullptr) {
    return AEE_ENOMEMORY;
  }
  const __fp16* xPtr = (const __fp16*)x;
  const __fp16* wPtr = (const __fp16*)w;
  const __fp16* rPtr = (const __fp16*)r;
  const __fp16* packedWPtr = (const __fp16*)packedW;
  const __fp16* packedRPtr = (const __fp16*)packedR;
  const __fp16* bPtr = (const __fp16*)b;
  const __fp16* h0Ptr = (const __fp16*)h0;
  const __fp16* c0Ptr = (const __fp16*)c0;
  __fp16* yPtr = (__fp16*)y;
  __fp16* yhPtr = (__fp16*)yh;
  __fp16* ycPtr = (__fp16*)yc;
  const int expectedPackedWeightBytes = np * weightKpStride * 1024 * (int)sizeof(__fp16);
  if (packedWeightBytes != expectedPackedWeightBytes) {
    packedWPtr = nullptr;
    packedRPtr = nullptr;
    packedWeightBytes = 0;
  }
  hmx_manager_enable_execution();
  hmx_unit_acquire();
  hmx_init_column_scales(vtcmScales, Q6_V_vsplat_R(0x3c00));
  hmx_set_output_scales(vtcmScales);

  for (int d = 0; d < direction; ++d) {
    const __fp16* wBase = wPtr + (size_t)d * gateSize * inputSize;
    const __fp16* rBase = rPtr + (size_t)d * gateSize * hiddenSize;
    const __fp16* packedWBase = packedWPtr != nullptr && packedWeightBytes > 0
                                    ? packedWPtr + (size_t)d * packedWeightBytes / sizeof(__fp16)
                                    : nullptr;
    const __fp16* packedRBase = packedRPtr != nullptr && packedWeightBytes > 0
                                    ? packedRPtr + (size_t)d * packedWeightBytes / sizeof(__fp16)
                                    : nullptr;
    const int bBase = d * biasDirectionStride;
    if (bBase + 4 * hiddenSize > bSize) {
      hmx_unit_release();
      hmx_manager_disable_execution();
      return AEE_EBADPARM;
    }
    const HVX_Vector zero_v = Q6_V_vzero();
    const HVX_Vector one_v = Q6_Vh_vsplat_R(0x3c00);
    const HVX_Vector two_v = Q6_Vh_vsplat_R(0x4000);
    if (packedWBase != nullptr && packedRBase != nullptr) {
      htp_ops_lstm_dma_copy_weight(vtcmInputWeight, packedWBase, packedWeightBytes);
      htp_ops_lstm_dma_copy_weight(vtcmRecurrentWeight, packedRBase, packedWeightBytes);
    } else {
      htp_ops_lstm_hmx_pack_weight_all(vtcmInputWeight, wBase, gateSize, inputSize, 1, inputSize, weightKpStride);
      htp_ops_lstm_hmx_pack_weight_all(vtcmRecurrentWeight, rBase, gateSize, hiddenSize, 1, hiddenSize,
                                       weightKpStride);
    }
    for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
      const int initIndex = (d * batch + batchIndex) * hiddenSize;
      if (initIndex < 0 || initIndex + hiddenSize > h0Size || initIndex + hiddenSize > c0Size) {
        hmx_unit_release();
        hmx_manager_disable_execution();
        return AEE_EBADPARM;
      }
      for (int h = 0; h < hiddenSize; h += 64) {
        const int stateIndex = batchIndex * hiddenSize + h;
        vmem((HVX_Vector*)(hidden + stateIndex)) = vmemu((const HVX_Vector*)(h0Ptr + initIndex + h));
        vmem((HVX_Vector*)(cell + stateIndex)) = vmemu((const HVX_Vector*)(c0Ptr + initIndex + h));
      }
    }

    const bool reverse = d == 1;
    HtpOpsLstmGateTask gateTaskStorage[8];
    worker_synctoken_t gateSyncToken;
    const int gateTasks = htp_ops_lstm_pick_gate_tasks(batch);
    HtpOpsLstmGateTask singleGateTask = {};
    if (gateTasks <= 1) {
      singleGateTask.recurrentGate = vtcmRecurrentGate;
      singleGateTask.hidden = hidden;
      singleGateTask.cell = cell;
      singleGateTask.y = yPtr;
      singleGateTask.biasBase = bPtr + bBase;
      singleGateTask.zero_v = zero_v;
      singleGateTask.one_v = one_v;
      singleGateTask.two_v = two_v;
      singleGateTask.hiddenSize = hiddenSize;
      singleGateTask.gateSize = gateSize;
      singleGateTask.packedGateTiles = np;
      singleGateTask.hiddenTiles = hiddenSize >> 5;
      singleGateTask.startBatch = 0;
      singleGateTask.batchCount = batch;
      singleGateTask.inputGateInVtcm = true;
      singleGateTask.recurrentGateInVtcm = true;
      singleGateTask.inputGatePacked = true;
      singleGateTask.recurrentGatePacked = true;
    } else {
      const int batchPerTask = (batch + gateTasks - 1) / gateTasks;
      for (int taskIndex = 0; taskIndex < gateTasks; ++taskIndex) {
        const int startBatch = taskIndex * batchPerTask;
        int endBatch = startBatch + batchPerTask;
        if (endBatch > batch) {
          endBatch = batch;
        }
        gateTaskStorage[taskIndex] = {};
        gateTaskStorage[taskIndex].recurrentGate = vtcmRecurrentGate;
        gateTaskStorage[taskIndex].hidden = hidden;
        gateTaskStorage[taskIndex].cell = cell;
        gateTaskStorage[taskIndex].y = yPtr;
        gateTaskStorage[taskIndex].biasBase = bPtr + bBase;
        gateTaskStorage[taskIndex].zero_v = zero_v;
        gateTaskStorage[taskIndex].one_v = one_v;
        gateTaskStorage[taskIndex].two_v = two_v;
        gateTaskStorage[taskIndex].hiddenSize = hiddenSize;
        gateTaskStorage[taskIndex].gateSize = gateSize;
        gateTaskStorage[taskIndex].packedGateTiles = np;
        gateTaskStorage[taskIndex].hiddenTiles = hiddenSize >> 5;
        gateTaskStorage[taskIndex].startBatch = startBatch;
        gateTaskStorage[taskIndex].batchCount = endBatch - startBatch;
        gateTaskStorage[taskIndex].inputGateInVtcm = true;
        gateTaskStorage[taskIndex].recurrentGateInVtcm = true;
        gateTaskStorage[taskIndex].inputGatePacked = true;
        gateTaskStorage[taskIndex].recurrentGatePacked = true;
        gateTaskStorage[taskIndex].sync_ctx = &gateSyncToken;
      }
    }
    worker_pool_job_t gateJob;
    gateJob.fptr = htp_ops_lstm_gate_worker;
    for (int blockStart = 0; blockStart < seqLength; blockStart += inputGateBlockSteps) {
      int blockSteps = seqLength - blockStart;
      if (blockSteps > inputGateBlockSteps) {
        blockSteps = inputGateBlockSteps;
      }
      const int copyBeginT = reverse ? seqLength - blockStart - blockSteps : blockStart;
      const __fp16* blockX = xPtr + (size_t)copyBeginT * batch * inputSize;
      if (!htp_ops_lstm_hmx_matmul_prepacked_k64_packed(vtcmStepInputGate, blockX, vtcmInputWeight, blockSteps * batch,
                                                        gateSize, inputKp, weightKpStride, inputSize,
                                                        vtcmActivation)) {
        hmx_unit_release();
        hmx_manager_disable_execution();
        return AEE_EFAILED;
      }
      for (int blockOffset = 0; blockOffset < blockSteps; ++blockOffset) {
        const int step = blockStart + blockOffset;
        const int t = reverse ? seqLength - 1 - step : step;
        if (!htp_ops_lstm_hmx_matmul_prepacked_k64_packed(vtcmRecurrentGate, hidden, vtcmRecurrentWeight, batch,
                                                          gateSize, recurrentKp, weightKpStride, hiddenSize,
                                                          vtcmActivation)) {
          hmx_unit_release();
          hmx_manager_disable_execution();
          return AEE_EFAILED;
        }
        const int yBaseIndex = ((t * direction + d) * batch) * hiddenSize;
        if (yBaseIndex < 0 || yBaseIndex + batch * hiddenSize > ySize) {
          hmx_unit_release();
          hmx_manager_disable_execution();
          return AEE_EBADPARM;
        }
        const int inputGateOffset = reverse ? blockSteps - 1 - blockOffset : blockOffset;
        if (gateTasks <= 1) {
          singleGateTask.inputGateBase = vtcmStepInputGate;
          singleGateTask.inputRowBase = inputGateOffset * batch;
          singleGateTask.yBaseIndex = yBaseIndex;
          htp_ops_lstm_gate_update_range(&singleGateTask);
        } else {
          HtpOpsLstmGateTask* tasks = gateTaskStorage;
          worker_synctoken_t* syncToken = &gateSyncToken;
          for (int taskIndex = 0; taskIndex < gateTasks; ++taskIndex) {
            tasks[taskIndex].inputGateBase = vtcmStepInputGate;
            tasks[taskIndex].inputRowBase = inputGateOffset * batch;
            tasks[taskIndex].yBaseIndex = yBaseIndex;
          }
          worker_pool_synctoken_init(syncToken, gateTasks);
          for (int taskIndex = 0; taskIndex < gateTasks; ++taskIndex) {
            gateJob.dptr = tasks + taskIndex;
            worker_pool_submit(NULL, gateJob);
          }
          worker_pool_synctoken_wait(syncToken);
        }
      }
    }
    if (outputCount > 1 && yhPtr != nullptr) {
      for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        const int base = d * stateSize + batchIndex * hiddenSize;
        if (base < 0 || base + hiddenSize > yhSize) {
          hmx_unit_release();
          hmx_manager_disable_execution();
          return AEE_EBADPARM;
        }
        for (int h = 0; h < hiddenSize; h += 64) {
          const int stateIndex = batchIndex * hiddenSize + h;
          vmemu((HVX_Vector*)(yhPtr + base + h)) = vmem((const HVX_Vector*)(hidden + stateIndex));
        }
      }
    }
    if (outputCount > 2 && ycPtr != nullptr) {
      for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        const int base = d * stateSize + batchIndex * hiddenSize;
        if (base < 0 || base + hiddenSize > ycSize) {
          hmx_unit_release();
          hmx_manager_disable_execution();
          return AEE_EBADPARM;
        }
        for (int h = 0; h < hiddenSize; h += 64) {
          const int stateIndex = batchIndex * hiddenSize + h;
          vmemu((HVX_Vector*)(ycPtr + base + h)) = vmem((const HVX_Vector*)(cell + stateIndex));
        }
      }
    }
  }

  hmx_unit_release();
  hmx_manager_disable_execution();
  (void)xSize;
  (void)wSize;
  (void)rSize;
  return AEE_SUCCESS;
}

AEEResult htp_ops_lstm(uint8_t* y, uint8_t* yh, uint8_t* yc, const uint8_t* x, const uint8_t* w,
                       const uint8_t* r, const uint8_t* b, const uint8_t* h0, const uint8_t* c0,
                       const uint8_t* packedW, const uint8_t* packedR,
                       int32_t seqLength, int32_t batch, int32_t inputSize, int32_t hiddenSize,
                       int32_t direction, int32_t bytes, int32_t outputCount, int32_t scratchBytes,
                       uint8_t* scratch, const int32_t* sizes, int32_t packedWeightBytes) {
  if (y == nullptr || x == nullptr || w == nullptr || r == nullptr || b == nullptr || h0 == nullptr || c0 == nullptr) {
    return AEE_EBADPARM;
  }
  if ((bytes != 2 && bytes != 4) || seqLength <= 0 || batch <= 0 || inputSize <= 0 || hiddenSize <= 0 ||
      direction <= 0 || sizes == nullptr) {
    return AEE_EBADPARM;
  }

  const int32_t xSize = sizes[0];
  const int32_t wSize = sizes[1];
  const int32_t rSize = sizes[2];
  const int32_t bSize = sizes[3];
  const int32_t h0Size = sizes[4];
  const int32_t c0Size = sizes[5];
  const int32_t ySize = sizes[6];
  const int32_t yhSize = sizes[7];
  const int32_t ycSize = sizes[8];
#define HTP_LSTM_CHECK(index, size) \
  do {                              \
    if ((index) < 0 || (index) >= (size)) return AEE_EBADPARM; \
  } while (0)

#if MNN_HEXAGON_LSTM_STAGE == 1
  for (int d = 0; d < direction; ++d) {
    const bool reverse = d == 1;
    for (int step = 0; step < seqLength; ++step) {
      const int t = reverse ? seqLength - 1 - step : step;
      for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        for (int h = 0; h < hiddenSize; ++h) {
          const int yIndex = ((t * direction + d) * batch + batchIndex) * hiddenSize + h;
          HTP_LSTM_CHECK(yIndex, ySize);
          htp_ops_lstm_write(y, yIndex, bytes, 0.0f);
        }
      }
    }
    if (outputCount > 1 && yh != nullptr) {
      for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        for (int h = 0; h < hiddenSize; ++h) {
          const int index = (d * batch + batchIndex) * hiddenSize + h;
          HTP_LSTM_CHECK(index, yhSize);
          htp_ops_lstm_write(yh, index, bytes, 0.0f);
        }
      }
    }
    if (outputCount > 2 && yc != nullptr) {
      for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        for (int h = 0; h < hiddenSize; ++h) {
          const int index = (d * batch + batchIndex) * hiddenSize + h;
          HTP_LSTM_CHECK(index, ycSize);
          htp_ops_lstm_write(yc, index, bytes, 0.0f);
        }
      }
    }
  }
  return AEE_SUCCESS;
#endif

  if (bytes == 2 && (inputSize & 63) == 0 && (hiddenSize & 63) == 0) {
    AEEResult fastRet = htp_ops_lstm_fast_fp16(y, yh, yc, x, w, r, b, h0, c0, packedW, packedR,
                                               seqLength, batch, inputSize, hiddenSize, direction, outputCount,
                                               scratch, scratchBytes, sizes, packedWeightBytes);
    if (fastRet != AEE_EUNSUPPORTED) {
      return fastRet;
    }
  }

  const int stateSize = batch * hiddenSize;
  const int biasDirectionStride = direction > 0 ? bSize / direction : 4 * hiddenSize;
  const size_t stateBytes = (size_t)stateSize * sizeof(float);
  const size_t requiredScratchBytes = 4 * stateBytes;
  if (scratch == nullptr || scratchBytes < 0 || (size_t)scratchBytes < requiredScratchBytes) {
    return AEE_ENOMEMORY;
  }
  uint8_t* scratchPtr = scratch;
  float* hidden = (float*)scratchPtr;
  scratchPtr += stateBytes;
  float* nextHidden = (float*)scratchPtr;
  scratchPtr += stateBytes;
  float* cell = (float*)scratchPtr;
  scratchPtr += stateBytes;
  float* nextCell = (float*)scratchPtr;

  const int gateSize = 4 * hiddenSize;
  for (int d = 0; d < direction; ++d) {
    for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
      for (int h = 0; h < hiddenSize; ++h) {
        const int stateIndex = batchIndex * hiddenSize + h;
        const int initIndex = (d * batch + batchIndex) * hiddenSize + h;
        HTP_LSTM_CHECK(initIndex, h0Size);
        HTP_LSTM_CHECK(initIndex, c0Size);
        hidden[stateIndex] = htp_ops_lstm_read(h0, initIndex, bytes);
        cell[stateIndex] = htp_ops_lstm_read(c0, initIndex, bytes);
      }
    }

    const bool reverse = d == 1;
    for (int step = 0; step < seqLength; ++step) {
      const int t = reverse ? seqLength - 1 - step : step;
      for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        for (int h = 0; h < hiddenSize; ++h) {
          const int bI = d * biasDirectionStride + h;
          const int bO = d * biasDirectionStride + hiddenSize + h;
          const int bF = d * biasDirectionStride + 2 * hiddenSize + h;
          const int bC = d * biasDirectionStride + 3 * hiddenSize + h;
          HTP_LSTM_CHECK(bI, bSize);
          HTP_LSTM_CHECK(bO, bSize);
          HTP_LSTM_CHECK(bF, bSize);
          HTP_LSTM_CHECK(bC, bSize);
          float gateI = htp_ops_lstm_read(b, bI, bytes);
          float gateO = htp_ops_lstm_read(b, bO, bytes);
          float gateF = htp_ops_lstm_read(b, bF, bytes);
          float gateC = htp_ops_lstm_read(b, bC, bytes);

          for (int i = 0; i < inputSize; ++i) {
            const int xIndex = (t * batch + batchIndex) * inputSize + i;
            HTP_LSTM_CHECK(xIndex, xSize);
            const float xv = htp_ops_lstm_read(x, xIndex, bytes);
            const int wBase = (d * gateSize) * inputSize + i;
            const int wI = wBase + h * inputSize;
            const int wO = wBase + (hiddenSize + h) * inputSize;
            const int wF = wBase + (2 * hiddenSize + h) * inputSize;
            const int wC = wBase + (3 * hiddenSize + h) * inputSize;
            HTP_LSTM_CHECK(wI, wSize);
            HTP_LSTM_CHECK(wO, wSize);
            HTP_LSTM_CHECK(wF, wSize);
            HTP_LSTM_CHECK(wC, wSize);
            gateI += htp_ops_lstm_read(w, wI, bytes) * xv;
            gateO += htp_ops_lstm_read(w, wO, bytes) * xv;
            gateF += htp_ops_lstm_read(w, wF, bytes) * xv;
            gateC += htp_ops_lstm_read(w, wC, bytes) * xv;
          }

          for (int rh = 0; rh < hiddenSize; ++rh) {
            const float hv = hidden[batchIndex * hiddenSize + rh];
            const int rBase = (d * gateSize) * hiddenSize + rh;
            const int rI = rBase + h * hiddenSize;
            const int rO = rBase + (hiddenSize + h) * hiddenSize;
            const int rF = rBase + (2 * hiddenSize + h) * hiddenSize;
            const int rC = rBase + (3 * hiddenSize + h) * hiddenSize;
            HTP_LSTM_CHECK(rI, rSize);
            HTP_LSTM_CHECK(rO, rSize);
            HTP_LSTM_CHECK(rF, rSize);
            HTP_LSTM_CHECK(rC, rSize);
            gateI += htp_ops_lstm_read(r, rI, bytes) * hv;
            gateO += htp_ops_lstm_read(r, rO, bytes) * hv;
            gateF += htp_ops_lstm_read(r, rF, bytes) * hv;
            gateC += htp_ops_lstm_read(r, rC, bytes) * hv;
          }

          const int stateIndex = batchIndex * hiddenSize + h;
          const float inputGate = htp_ops_lstm_sigmoid(gateI);
          const float outputGate = htp_ops_lstm_sigmoid(gateO);
          const float forgetGate = htp_ops_lstm_sigmoid(gateF);
          const float cellGate = htp_ops_lstm_tanh(gateC);
          const float cellValue = forgetGate * cell[stateIndex] + inputGate * cellGate;
          nextCell[stateIndex] = cellValue;
          const float hiddenValue = outputGate * htp_ops_lstm_tanh(cellValue);
          nextHidden[stateIndex] = hiddenValue;
          const int yIndex = ((t * direction + d) * batch + batchIndex) * hiddenSize + h;
          HTP_LSTM_CHECK(yIndex, ySize);
          htp_ops_lstm_write(y, yIndex, bytes, hiddenValue);
        }
      }
      float* hiddenTmp = hidden;
      hidden = nextHidden;
      nextHidden = hiddenTmp;
      float* tmp = cell;
      cell = nextCell;
      nextCell = tmp;
    }

    if (outputCount > 1 && yh != nullptr) {
      for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        for (int h = 0; h < hiddenSize; ++h) {
          const int index = (d * batch + batchIndex) * hiddenSize + h;
          HTP_LSTM_CHECK(index, yhSize);
          htp_ops_lstm_write(yh, index, bytes, hidden[batchIndex * hiddenSize + h]);
        }
      }
    }
    if (outputCount > 2 && yc != nullptr) {
      for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        for (int h = 0; h < hiddenSize; ++h) {
          const int index = (d * batch + batchIndex) * hiddenSize + h;
          HTP_LSTM_CHECK(index, ycSize);
          htp_ops_lstm_write(yc, index, bytes, cell[batchIndex * hiddenSize + h]);
        }
      }
    }
  }

#undef HTP_LSTM_CHECK
  return AEE_SUCCESS;
}

} // extern "C"

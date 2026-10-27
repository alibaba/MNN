#ifndef MNN_HTP_GATED_RMS_NORM_PARAMS_H
#define MNN_HTP_GATED_RMS_NORM_PARAMS_H

// GatedRMSNorm shaping, shared by the host (the geometry gate in GeometryGatedRMSNorm.cpp and the
// creator in execution/HexagonGatedRMSNorm.cpp) and the DSP (the kernel in linear_attention_ops.cc).
//
// The geometry has to decide whether to keep the op whole *before* any creator runs, and if it keeps an
// op whole that the creator then refuses there is nothing left to pick it up -- no backend has a
// GatedRMSNorm execution to fall back to, so the session fails outright. The two must therefore agree,
// which is why the constant lives here instead of being written twice.
//
// Host-includable on purpose: plain C, no HVX types, no DSP headers. Same contract as htp_attn_params.h.

// Channels the kernel handles per block; `inside` must be a multiple of it.
#define HTP_GATED_RMS_NORM_PACK 64

#endif  // MNN_HTP_GATED_RMS_NORM_PARAMS_H

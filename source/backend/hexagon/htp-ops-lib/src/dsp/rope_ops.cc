#include "dsp/mmap_mgr.h"
#include "dsp/vtcm_mgr.h"
#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <qurt_memory.h>
#include <stdint.h>
#include <math.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#include <remote.h>
#include "htp_ops.h"
#include "dsp/hvx_utils.h"
#include "dsp/worker_pool.h"

extern "C" {


#define PREFETCH_SIZE   (8 * 1024)
#define PREFETCH_N_VECS (PREFETCH_SIZE / VLEN)

typedef struct {
    worker_synctoken_t sync_ctx;
    unsigned int       task_id;
    int                n_tasks;
    int                batch_seq;
    int                num_head;
    int                kv_num_head;
    int                head_dim;
    int                half_head_dim;
    int                rope_half_head_dim;
    int                rope_dim;
    float              q_epsilon;
    float              k_epsilon;
    int                q_rms_norm;
    int                k_rms_norm;
    size_t             worker_vtcm_bytes;
    uint8_t*           workspace_base;
    const __fp16*      q_in;
    __fp16*            q_out;
    const __fp16*      k_in;
    __fp16*            k_out;
    const __fp16*      cos_even;
    const __fp16*      cos_odd;
    const __fp16*      sin_even;
    const __fp16*      sin_odd;
    const float*       q_gamma;
    const float*       k_gamma;
    int                input_c4;
} HtpOpsRopeFuseLayernormBatchTaskState;

typedef struct {
    worker_synctoken_t sync_ctx;
    unsigned int       task_id;
    int                batch_seq;
    int                num_head;
    int                kv_num_head;
    int                head_dim;
    int                half_head_dim;
    int                rope_half_head_dim;
    int                rope_dim;
    const __fp16*      q_in;
    __fp16*            q_out;
    const __fp16*      k_in;
    __fp16*            k_out;
    const __fp16*      cos_even;
    const __fp16*      cos_odd;
    const __fp16*      sin_even;
    const __fp16*      sin_odd;
    int                input_c4;
} HtpOpsRopeBatchTaskState;

static inline void compute_rope_head_fp16(__fp16* out_base, const __fp16* in_base, int num_head, int head_dim,
                                          const __fp16* c_e, const __fp16* c_o,
                                          const __fp16* s_e, const __fp16* s_o,
                                          int rope_half_head_dim, int half_head_dim);
static inline void compute_layernorm_head_fp16(__fp16* out_base, const __fp16* in_base, const float* gamma,
                                               int headNumber, int head_dim, float epsilon, int rms_norm);

static inline void htp_ops_rope_unpack_c4_token(const __fp16* src, __fp16* dst, int token, int seq_len, int channel) {
    const int pack = 64;
    int c = 0;
    for (; c + pack <= channel; c += pack) {
        const __fp16* src_pack = src + (size_t)(c / pack) * seq_len * pack + (size_t)token * pack;
        vmemu((HVX_Vector*)(dst + c)) = vmemu((const HVX_Vector*)src_pack);
    }
    if (c < channel) {
        const __fp16* src_pack = src + (size_t)(c / pack) * seq_len * pack + (size_t)token * pack;
        memcpy(dst + c, src_pack, (size_t)(channel - c) * sizeof(__fp16));
    }
}

static inline const __fp16* htp_ops_rope_c4_pack_ptr(const __fp16* src, int token, int seq_len, int channel) {
    const int pack = 64;
    return src + (size_t)(channel / pack) * seq_len * pack + (size_t)token * pack + (channel % pack);
}

static inline void htp_ops_rope_load_trig64(const __fp16* c_e, const __fp16* c_o,
                                            const __fp16* s_e, const __fp16* s_o,
                                            int lane_count,
                                            HVX_Vector* vc0, HVX_Vector* vc1,
                                            HVX_Vector* vse, HVX_Vector* vso) {
    if (lane_count >= 64) {
        *vc0 = vmemu((HVX_Vector*)c_e);
        *vc1 = vmemu((HVX_Vector*)c_o);
        *vse = vmemu((HVX_Vector*)s_e);
        *vso = vmemu((HVX_Vector*)s_o);
        return;
    }
    __fp16 tmp_c0[64] __attribute__((aligned(128)));
    __fp16 tmp_c1[64] __attribute__((aligned(128)));
    __fp16 tmp_s0[64] __attribute__((aligned(128)));
    __fp16 tmp_s1[64] __attribute__((aligned(128)));
    int i = 0;
    for (; i < lane_count; ++i) {
        tmp_c0[i] = c_e[i];
        tmp_c1[i] = c_o[i];
        tmp_s0[i] = s_e[i];
        tmp_s1[i] = s_o[i];
    }
    for (; i < 64; ++i) {
        tmp_c0[i] = (__fp16)1.0f;
        tmp_c1[i] = (__fp16)1.0f;
        tmp_s0[i] = (__fp16)0.0f;
        tmp_s1[i] = (__fp16)0.0f;
    }
    *vc0 = vmem((HVX_Vector*)tmp_c0);
    *vc1 = vmem((HVX_Vector*)tmp_c1);
    *vse = vmem((HVX_Vector*)tmp_s0);
    *vso = vmem((HVX_Vector*)tmp_s1);
}

static inline void htp_ops_rope_apply64(HVX_Vector v_q0, HVX_Vector v_q1,
                                        HVX_Vector v_c0, HVX_Vector v_c1,
                                        HVX_Vector v_se, HVX_Vector v_so,
                                        HVX_Vector* v_res0, HVX_Vector* v_res1) {
    *v_res0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_Vqf16Vqf16(
        Q6_Vqf16_vmpy_VhfVhf(v_q0, v_c0), Q6_Vqf16_vmpy_VhfVhf(v_q1, v_se)));
    *v_res1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(
        Q6_Vqf16_vmpy_VhfVhf(v_q1, v_c1), Q6_Vqf16_vmpy_VhfVhf(v_q0, v_so)));
}

static inline void compute_rope_head64blocks_fp16(__fp16* out_base, const __fp16* in_base,
                                                  int num_head, int head_dim, int half_head_dim,
                                                  const __fp16* c_e, const __fp16* c_o,
                                                  const __fp16* s_e, const __fp16* s_o,
                                                  int rope_half_head_dim) {
    for (int d = 0; d < half_head_dim; d += 64) {
        int lane_count = rope_half_head_dim - d;
        if (lane_count > 64) {
            lane_count = 64;
        } else if (lane_count < 0) {
            lane_count = 0;
        }

        if (lane_count == 0) {
            int h = 0;
            for (; h + 2 <= num_head; h += 2) {
                const __fp16* in_h0 = in_base + (size_t)h * head_dim;
                const __fp16* in_h1 = in_h0 + head_dim;
                __fp16* out_h0 = out_base + (size_t)h * head_dim;
                __fp16* out_h1 = out_h0 + head_dim;
                vmemu((HVX_Vector*)(out_h0 + d)) = vmemu((HVX_Vector*)(in_h0 + d));
                vmemu((HVX_Vector*)(out_h0 + half_head_dim + d)) = vmemu((HVX_Vector*)(in_h0 + half_head_dim + d));
                vmemu((HVX_Vector*)(out_h1 + d)) = vmemu((HVX_Vector*)(in_h1 + d));
                vmemu((HVX_Vector*)(out_h1 + half_head_dim + d)) = vmemu((HVX_Vector*)(in_h1 + half_head_dim + d));
            }
            for (; h < num_head; ++h) {
                const __fp16* in_h = in_base + (size_t)h * head_dim;
                __fp16* out_h = out_base + (size_t)h * head_dim;
                vmemu((HVX_Vector*)(out_h + d)) = vmemu((HVX_Vector*)(in_h + d));
                vmemu((HVX_Vector*)(out_h + half_head_dim + d)) = vmemu((HVX_Vector*)(in_h + half_head_dim + d));
            }
            continue;
        }

        HVX_Vector v_c0, v_c1, v_se, v_so;
        htp_ops_rope_load_trig64(c_e + d, c_o + d, s_e + d, s_o + d,
                                 lane_count, &v_c0, &v_c1, &v_se, &v_so);
        int h = 0;
        for (; h + 2 <= num_head; h += 2) {
            const __fp16* in_h0 = in_base + (size_t)h * head_dim;
            const __fp16* in_h1 = in_h0 + head_dim;
            __fp16* out_h0 = out_base + (size_t)h * head_dim;
            __fp16* out_h1 = out_h0 + head_dim;

            HVX_Vector v_q00 = vmemu((HVX_Vector*)(in_h0 + d));
            HVX_Vector v_q01 = vmemu((HVX_Vector*)(in_h0 + half_head_dim + d));
            HVX_Vector v_q10 = vmemu((HVX_Vector*)(in_h1 + d));
            HVX_Vector v_q11 = vmemu((HVX_Vector*)(in_h1 + half_head_dim + d));
            HVX_Vector v_res00, v_res01, v_res10, v_res11;
            htp_ops_rope_apply64(v_q00, v_q01, v_c0, v_c1, v_se, v_so, &v_res00, &v_res01);
            htp_ops_rope_apply64(v_q10, v_q11, v_c0, v_c1, v_se, v_so, &v_res10, &v_res11);

            vmemu((HVX_Vector*)(out_h0 + d)) = v_res00;
            vmemu((HVX_Vector*)(out_h0 + half_head_dim + d)) = v_res01;
            vmemu((HVX_Vector*)(out_h1 + d)) = v_res10;
            vmemu((HVX_Vector*)(out_h1 + half_head_dim + d)) = v_res11;
        }
        for (; h < num_head; ++h) {
            const __fp16* in_h = in_base + (size_t)h * head_dim;
            __fp16* out_h = out_base + (size_t)h * head_dim;
            HVX_Vector v_q0 = vmemu((HVX_Vector*)(in_h + d));
            HVX_Vector v_q1 = vmemu((HVX_Vector*)(in_h + half_head_dim + d));
            HVX_Vector v_res0, v_res1;
            htp_ops_rope_apply64(v_q0, v_q1, v_c0, v_c1, v_se, v_so, &v_res0, &v_res1);
            vmemu((HVX_Vector*)(out_h + d)) = v_res0;
            vmemu((HVX_Vector*)(out_h + half_head_dim + d)) = v_res1;
        }
    }
}

static inline void compute_rope_head64blocks_c4_fp16(__fp16* out_base, const __fp16* in_c4,
                                                     int token, int seq_len, int num_head,
                                                     int head_dim, int half_head_dim,
                                                     const __fp16* c_e, const __fp16* c_o,
                                                     const __fp16* s_e, const __fp16* s_o,
                                                     int rope_half_head_dim) {
    int d = 0;
    for (; d + 64 <= half_head_dim; d += 64) {
        int lane_count = rope_half_head_dim - d;
        if (lane_count > 64) {
            lane_count = 64;
        } else if (lane_count < 0) {
            lane_count = 0;
        }

        if (lane_count == 0) {
            int h = 0;
            for (; h + 2 <= num_head; h += 2) {
                const int c0 = h * head_dim;
                const int c1 = c0 + head_dim;
                __fp16* out_h0 = out_base + (size_t)h * head_dim;
                __fp16* out_h1 = out_h0 + head_dim;
                vmemu((HVX_Vector*)(out_h0 + d)) =
                    vmemu((HVX_Vector*)htp_ops_rope_c4_pack_ptr(in_c4, token, seq_len, c0 + d));
                vmemu((HVX_Vector*)(out_h0 + half_head_dim + d)) =
                    vmemu((HVX_Vector*)htp_ops_rope_c4_pack_ptr(in_c4, token, seq_len, c0 + half_head_dim + d));
                vmemu((HVX_Vector*)(out_h1 + d)) =
                    vmemu((HVX_Vector*)htp_ops_rope_c4_pack_ptr(in_c4, token, seq_len, c1 + d));
                vmemu((HVX_Vector*)(out_h1 + half_head_dim + d)) =
                    vmemu((HVX_Vector*)htp_ops_rope_c4_pack_ptr(in_c4, token, seq_len, c1 + half_head_dim + d));
            }
            for (; h < num_head; ++h) {
                const int c = h * head_dim;
                __fp16* out_h = out_base + (size_t)h * head_dim;
                vmemu((HVX_Vector*)(out_h + d)) =
                    vmemu((HVX_Vector*)htp_ops_rope_c4_pack_ptr(in_c4, token, seq_len, c + d));
                vmemu((HVX_Vector*)(out_h + half_head_dim + d)) =
                    vmemu((HVX_Vector*)htp_ops_rope_c4_pack_ptr(in_c4, token, seq_len, c + half_head_dim + d));
            }
            continue;
        }

        HVX_Vector v_c0, v_c1, v_se, v_so;
        htp_ops_rope_load_trig64(c_e + d, c_o + d, s_e + d, s_o + d,
                                 lane_count, &v_c0, &v_c1, &v_se, &v_so);
        int h = 0;
        for (; h + 2 <= num_head; h += 2) {
            const int c0 = h * head_dim;
            const int c1 = c0 + head_dim;
            __fp16* out_h0 = out_base + (size_t)h * head_dim;
            __fp16* out_h1 = out_h0 + head_dim;

            HVX_Vector v_q00 = vmemu((HVX_Vector*)htp_ops_rope_c4_pack_ptr(in_c4, token, seq_len, c0 + d));
            HVX_Vector v_q01 = vmemu((HVX_Vector*)htp_ops_rope_c4_pack_ptr(in_c4, token, seq_len,
                                                                           c0 + half_head_dim + d));
            HVX_Vector v_q10 = vmemu((HVX_Vector*)htp_ops_rope_c4_pack_ptr(in_c4, token, seq_len, c1 + d));
            HVX_Vector v_q11 = vmemu((HVX_Vector*)htp_ops_rope_c4_pack_ptr(in_c4, token, seq_len,
                                                                           c1 + half_head_dim + d));
            HVX_Vector v_res00, v_res01, v_res10, v_res11;
            htp_ops_rope_apply64(v_q00, v_q01, v_c0, v_c1, v_se, v_so, &v_res00, &v_res01);
            htp_ops_rope_apply64(v_q10, v_q11, v_c0, v_c1, v_se, v_so, &v_res10, &v_res11);

            vmemu((HVX_Vector*)(out_h0 + d)) = v_res00;
            vmemu((HVX_Vector*)(out_h0 + half_head_dim + d)) = v_res01;
            vmemu((HVX_Vector*)(out_h1 + d)) = v_res10;
            vmemu((HVX_Vector*)(out_h1 + half_head_dim + d)) = v_res11;
        }
        for (; h < num_head; ++h) {
            const int c = h * head_dim;
            __fp16* out_h = out_base + (size_t)h * head_dim;
            HVX_Vector v_q0 = vmemu((HVX_Vector*)htp_ops_rope_c4_pack_ptr(in_c4, token, seq_len, c + d));
            HVX_Vector v_q1 = vmemu((HVX_Vector*)htp_ops_rope_c4_pack_ptr(in_c4, token, seq_len,
                                                                          c + half_head_dim + d));
            HVX_Vector v_res0, v_res1;
            htp_ops_rope_apply64(v_q0, v_q1, v_c0, v_c1, v_se, v_so, &v_res0, &v_res1);
            vmemu((HVX_Vector*)(out_h + d)) = v_res0;
            vmemu((HVX_Vector*)(out_h + half_head_dim + d)) = v_res1;
        }
    }

    if (d < half_head_dim) {
        int lane_count = rope_half_head_dim - d;
        if (lane_count > 32) {
            lane_count = 32;
        } else if (lane_count < 0) {
            lane_count = 0;
        }
        HVX_Vector v_c0, v_c1, v_se, v_so;
        htp_ops_rope_load_trig64(c_e + d, c_o + d, s_e + d, s_o + d,
                                 lane_count, &v_c0, &v_c1, &v_se, &v_so);
        for (int h = 0; h < num_head; ++h) {
            const int c = h * head_dim;
            __fp16* out_h = out_base + (size_t)h * head_dim;
            HVX_Vector v_q0 = vmemu((HVX_Vector*)htp_ops_rope_c4_pack_ptr(in_c4, token, seq_len, c + d));
            HVX_Vector v_q1 = vmemu((HVX_Vector*)htp_ops_rope_c4_pack_ptr(in_c4, token, seq_len,
                                                                          c + half_head_dim + d));
            HVX_Vector v_res0, v_res1;
            htp_ops_rope_apply64(v_q0, v_q1, v_c0, v_c1, v_se, v_so, &v_res0, &v_res1);
            vstu_variable(out_h + d, 32 * sizeof(__fp16), v_res0);
            vstu_variable(out_h + half_head_dim + d, 32 * sizeof(__fp16), v_res1);
        }
    }
}

static inline size_t htp_ops_rope_align_vtcm_bytes(size_t size) {
    return (size + 127) & ~((size_t)127);
}

static inline float htp_ops_rope_reduce_sum2_f32(HVX_Vector acc0, HVX_Vector acc1) {
    HVX_Vector v = Q6_Vsf_vadd_VsfVsf(acc0, acc1);
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 4));
    union { HVX_Vector v; float f[32]; } u = { .v = v };
    return u.f[0];
}

static inline float htp_ops_rope_fast_rsqrtf(float x) {
    union { float f; int32_t i; } u = { .f = x };
    float half_x = 0.5f * x;
    u.i = 0x5f3759df - (u.i >> 1);
    float y = u.f;
    y = y * (1.5f - half_x * y * y);
    return y;
}

static inline HVX_Vector htp_ops_norm128_half_fp16(HVX_Vector x_hf, HVX_Vector vmean, HVX_Vector vinv_std,
                                                   HVX_Vector gamma0, HVX_Vector gamma1, int rms_norm) {
    HVX_VectorPair x_sf = Q6_Wsf_vcvt_Vhf(x_hf);
    HVX_Vector out0 = Q6_V_lo_W(x_sf);
    HVX_Vector out1 = Q6_V_hi_W(x_sf);
    if (!rms_norm) {
        out0 = Q6_Vsf_vsub_VsfVsf(out0, vmean);
        out1 = Q6_Vsf_vsub_VsfVsf(out1, vmean);
    }
    out0 = Q6_Vsf_vmpy_VsfVsf(out0, vinv_std);
    out1 = Q6_Vsf_vmpy_VsfVsf(out1, vinv_std);
    out0 = Q6_Vsf_vmpy_VsfVsf(out0, gamma0);
    out1 = Q6_Vsf_vmpy_VsfVsf(out1, gamma1);
    return Q6_Vhf_vcvt_VsfVsf(out0, out1);
}

static inline void htp_ops_store_norm_rope128(__fp16* out_h, HVX_Vector x0_hf, HVX_Vector x1_hf,
                                              HVX_Vector vmean, HVX_Vector vinv_std,
                                              HVX_Vector g00, HVX_Vector g01,
                                              HVX_Vector g10, HVX_Vector g11,
                                              HVX_Vector vc0, HVX_Vector vc1,
                                              HVX_Vector vse, HVX_Vector vso, int rms_norm) {
    HVX_Vector n0_hf = htp_ops_norm128_half_fp16(x0_hf, vmean, vinv_std, g00, g01, rms_norm);
    HVX_Vector n1_hf = htp_ops_norm128_half_fp16(x1_hf, vmean, vinv_std, g10, g11, rms_norm);
    HVX_Vector out0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_Vqf16Vqf16(Q6_Vqf16_vmpy_VhfVhf(n0_hf, vc0),
                                                                    Q6_Vqf16_vmpy_VhfVhf(n1_hf, vse)));
    HVX_Vector out1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(Q6_Vqf16_vmpy_VhfVhf(n1_hf, vc1),
                                                                    Q6_Vqf16_vmpy_VhfVhf(n0_hf, vso)));
    vmemu((HVX_Vector*)out_h) = out0;
    vmemu((HVX_Vector*)(out_h + 64)) = out1;
}

static inline void htp_ops_store_rmsnorm_rope128_hscale(__fp16* out_h, HVX_Vector x0_hf, HVX_Vector x1_hf,
                                                        HVX_Vector vinv_std,
                                                        HVX_Vector g00, HVX_Vector g01,
                                                        HVX_Vector g10, HVX_Vector g11,
                                                        HVX_Vector vc0, HVX_Vector vc1,
                                                        HVX_Vector vse, HVX_Vector vso) {
    HVX_Vector scale0 = Q6_Vhf_vcvt_VsfVsf(Q6_Vsf_vmpy_VsfVsf(g00, vinv_std),
                                            Q6_Vsf_vmpy_VsfVsf(g01, vinv_std));
    HVX_Vector scale1 = Q6_Vhf_vcvt_VsfVsf(Q6_Vsf_vmpy_VsfVsf(g10, vinv_std),
                                            Q6_Vsf_vmpy_VsfVsf(g11, vinv_std));
    HVX_Vector n0_hf = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(x0_hf, scale0));
    HVX_Vector n1_hf = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(x1_hf, scale1));
    HVX_Vector out0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_Vqf16Vqf16(Q6_Vqf16_vmpy_VhfVhf(n0_hf, vc0),
                                                                    Q6_Vqf16_vmpy_VhfVhf(n1_hf, vse)));
    HVX_Vector out1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(Q6_Vqf16_vmpy_VhfVhf(n1_hf, vc1),
                                                                    Q6_Vqf16_vmpy_VhfVhf(n0_hf, vso)));
    vmemu((HVX_Vector*)out_h) = out0;
    vmemu((HVX_Vector*)(out_h + 64)) = out1;
}

static inline void compute_rmsnorm_rope_one_head128_fp16(__fp16* out_h, const __fp16* in_h,
                                                         HVX_Vector g00, HVX_Vector g01,
                                                         HVX_Vector g10, HVX_Vector g11,
                                                         const __fp16* c_e, const __fp16* c_o,
                                                         const __fp16* s_e, const __fp16* s_o,
                                                         float epsilon) {
    HVX_Vector x0_hf = *((const HVX_UVector*)in_h);
    HVX_Vector x1_hf = *((const HVX_UVector*)(in_h + 64));
    HVX_VectorPair x0_sf = Q6_Wsf_vcvt_Vhf(x0_hf);
    HVX_VectorPair x1_sf = Q6_Wsf_vcvt_Vhf(x1_hf);
    HVX_Vector x00 = Q6_V_lo_W(x0_sf);
    HVX_Vector x01 = Q6_V_hi_W(x0_sf);
    HVX_Vector x10 = Q6_V_lo_W(x1_sf);
    HVX_Vector x11 = Q6_V_hi_W(x1_sf);

    HVX_Vector vsqsum0 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(x00, x00),
                                             Q6_Vsf_vmpy_VsfVsf(x10, x10));
    HVX_Vector vsqsum1 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(x01, x01),
                                             Q6_Vsf_vmpy_VsfVsf(x11, x11));

    float var = htp_ops_rope_reduce_sum2_f32(vsqsum0, vsqsum1) / 128.0f;
    float inv_std = htp_ops_rope_fast_rsqrtf(var + epsilon);

    union { float f; int32_t i; } u_inv_std = { .f = inv_std };
    HVX_Vector vinv_std = Q6_V_vsplat_R(u_inv_std.i);

    HVX_Vector vc0 = vmemu((HVX_Vector*)c_e);
    HVX_Vector vc1 = vmemu((HVX_Vector*)c_o);
    HVX_Vector vse = vmemu((HVX_Vector*)s_e);
    HVX_Vector vso = vmemu((HVX_Vector*)s_o);
    htp_ops_store_rmsnorm_rope128_hscale(out_h, x0_hf, x1_hf, vinv_std,
                                         g00, g01, g10, g11, vc0, vc1, vse, vso);
}

static inline void compute_rmsnorm_rope_two_heads128_fp16(__fp16* out0_h, __fp16* out1_h,
                                                          const __fp16* in0_h, const __fp16* in1_h,
                                                          HVX_Vector g00, HVX_Vector g01,
                                                          HVX_Vector g10, HVX_Vector g11,
                                                          const __fp16* c_e, const __fp16* c_o,
                                                          const __fp16* s_e, const __fp16* s_o,
                                                          float epsilon) {
    HVX_Vector x00_hf = *((const HVX_UVector*)in0_h);
    HVX_Vector x01_hf = *((const HVX_UVector*)(in0_h + 64));
    HVX_Vector x10_hf = *((const HVX_UVector*)in1_h);
    HVX_Vector x11_hf = *((const HVX_UVector*)(in1_h + 64));

    HVX_VectorPair x00_sf = Q6_Wsf_vcvt_Vhf(x00_hf);
    HVX_VectorPair x01_sf = Q6_Wsf_vcvt_Vhf(x01_hf);
    HVX_VectorPair x10_sf = Q6_Wsf_vcvt_Vhf(x10_hf);
    HVX_VectorPair x11_sf = Q6_Wsf_vcvt_Vhf(x11_hf);

    HVX_Vector h0_sq0 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(x00_sf), Q6_V_lo_W(x00_sf)),
                                           Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(x01_sf), Q6_V_lo_W(x01_sf)));
    HVX_Vector h0_sq1 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(x00_sf), Q6_V_hi_W(x00_sf)),
                                           Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(x01_sf), Q6_V_hi_W(x01_sf)));
    HVX_Vector h1_sq0 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(x10_sf), Q6_V_lo_W(x10_sf)),
                                           Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(x11_sf), Q6_V_lo_W(x11_sf)));
    HVX_Vector h1_sq1 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(x10_sf), Q6_V_hi_W(x10_sf)),
                                           Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(x11_sf), Q6_V_hi_W(x11_sf)));

    float var0 = htp_ops_rope_reduce_sum2_f32(h0_sq0, h0_sq1) / 128.0f;
    float var1 = htp_ops_rope_reduce_sum2_f32(h1_sq0, h1_sq1) / 128.0f;
    float inv_std0 = htp_ops_rope_fast_rsqrtf(var0 + epsilon);
    float inv_std1 = htp_ops_rope_fast_rsqrtf(var1 + epsilon);

    union { float f; int32_t i; } u_inv_std0 = { .f = inv_std0 };
    union { float f; int32_t i; } u_inv_std1 = { .f = inv_std1 };
    HVX_Vector vinv_std0 = Q6_V_vsplat_R(u_inv_std0.i);
    HVX_Vector vinv_std1 = Q6_V_vsplat_R(u_inv_std1.i);

    HVX_Vector vc0 = vmemu((HVX_Vector*)c_e);
    HVX_Vector vc1 = vmemu((HVX_Vector*)c_o);
    HVX_Vector vse = vmemu((HVX_Vector*)s_e);
    HVX_Vector vso = vmemu((HVX_Vector*)s_o);

    htp_ops_store_rmsnorm_rope128_hscale(out0_h, x00_hf, x01_hf, vinv_std0,
                                         g00, g01, g10, g11, vc0, vc1, vse, vso);
    htp_ops_store_rmsnorm_rope128_hscale(out1_h, x10_hf, x11_hf, vinv_std1,
                                         g00, g01, g10, g11, vc0, vc1, vse, vso);
}

static inline void compute_rmsnorm_rope_head128_fp16(__fp16* out_base, const __fp16* in_base,
                                                     const float* gamma, int headNumber,
                                                     const __fp16* c_e, const __fp16* c_o,
                                                     const __fp16* s_e, const __fp16* s_o,
                                                     float epsilon) {
    HVX_Vector vg00 = *((const HVX_UVector*)gamma);
    HVX_Vector vg01 = *((const HVX_UVector*)(gamma + 32));
    HVX_VectorPair g0_deal = Q6_W_vdeal_VVR(vg01, vg00, -4);
    HVX_Vector vg10 = *((const HVX_UVector*)(gamma + 64));
    HVX_Vector vg11 = *((const HVX_UVector*)(gamma + 96));
    HVX_VectorPair g1_deal = Q6_W_vdeal_VVR(vg11, vg10, -4);
    HVX_Vector g00 = Q6_V_lo_W(g0_deal);
    HVX_Vector g01 = Q6_V_hi_W(g0_deal);
    HVX_Vector g10 = Q6_V_lo_W(g1_deal);
    HVX_Vector g11 = Q6_V_hi_W(g1_deal);
    int h = 0;
    for (; h + 2 <= headNumber; h += 2) {
        compute_rmsnorm_rope_two_heads128_fp16(out_base + (size_t)h * 128,
                                               out_base + (size_t)(h + 1) * 128,
                                               in_base + (size_t)h * 128,
                                               in_base + (size_t)(h + 1) * 128,
                                               g00, g01, g10, g11, c_e, c_o, s_e, s_o, epsilon);
    }
    for (; h < headNumber; ++h) {
        compute_rmsnorm_rope_one_head128_fp16(out_base + (size_t)h * 128,
                                              in_base + (size_t)h * 128,
                                              g00, g01, g10, g11, c_e, c_o, s_e, s_o, epsilon);
    }
}

static inline void compute_layernorm_rope_one_head128_fp16(__fp16* out_h, const __fp16* in_h,
                                                           const float* gamma, const __fp16* c_e,
                                                           const __fp16* c_o, const __fp16* s_e,
                                                           const __fp16* s_o, float epsilon, int rms_norm) {
    HVX_Vector x0_hf = *((const HVX_UVector*)in_h);
    HVX_Vector x1_hf = *((const HVX_UVector*)(in_h + 64));
    HVX_VectorPair x0_sf = Q6_Wsf_vcvt_Vhf(x0_hf);
    HVX_VectorPair x1_sf = Q6_Wsf_vcvt_Vhf(x1_hf);
    HVX_Vector x00 = Q6_V_lo_W(x0_sf);
    HVX_Vector x01 = Q6_V_hi_W(x0_sf);
    HVX_Vector x10 = Q6_V_lo_W(x1_sf);
    HVX_Vector x11 = Q6_V_hi_W(x1_sf);

    HVX_Vector vsum0 = Q6_Vsf_vadd_VsfVsf(x00, x10);
    HVX_Vector vsum1 = Q6_Vsf_vadd_VsfVsf(x01, x11);
    HVX_Vector vsqsum0 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(x00, x00),
                                             Q6_Vsf_vmpy_VsfVsf(x10, x10));
    HVX_Vector vsqsum1 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(x01, x01),
                                             Q6_Vsf_vmpy_VsfVsf(x11, x11));

    float mean = 0.0f;
    if (!rms_norm) {
        mean = htp_ops_rope_reduce_sum2_f32(vsum0, vsum1) / 128.0f;
    }
    float var = htp_ops_rope_reduce_sum2_f32(vsqsum0, vsqsum1) / 128.0f;
    if (!rms_norm) {
        var -= mean * mean;
    }
    float inv_std = htp_ops_rope_fast_rsqrtf(var + epsilon);

    union { float f; int32_t i; } u_mean = { .f = mean };
    union { float f; int32_t i; } u_inv_std = { .f = inv_std };
    HVX_Vector vmean = Q6_V_vsplat_R(u_mean.i);
    HVX_Vector vinv_std = Q6_V_vsplat_R(u_inv_std.i);

    HVX_Vector vg00 = *((const HVX_UVector*)gamma);
    HVX_Vector vg01 = *((const HVX_UVector*)(gamma + 32));
    HVX_VectorPair g0_deal = Q6_W_vdeal_VVR(vg01, vg00, -4);
    HVX_Vector vg10 = *((const HVX_UVector*)(gamma + 64));
    HVX_Vector vg11 = *((const HVX_UVector*)(gamma + 96));
    HVX_VectorPair g1_deal = Q6_W_vdeal_VVR(vg11, vg10, -4);

    HVX_Vector vc0 = vmemu((HVX_Vector*)c_e);
    HVX_Vector vc1 = vmemu((HVX_Vector*)c_o);
    HVX_Vector vse = vmemu((HVX_Vector*)s_e);
    HVX_Vector vso = vmemu((HVX_Vector*)s_o);
    htp_ops_store_norm_rope128(out_h, x0_hf, x1_hf, vmean, vinv_std,
                               Q6_V_lo_W(g0_deal), Q6_V_hi_W(g0_deal),
                               Q6_V_lo_W(g1_deal), Q6_V_hi_W(g1_deal),
                               vc0, vc1, vse, vso, rms_norm);
}

static inline void compute_layernorm_rope_two_heads128_fp16(__fp16* out0_h, __fp16* out1_h,
                                                            const __fp16* in0_h, const __fp16* in1_h,
                                                            const float* gamma, const __fp16* c_e,
                                                            const __fp16* c_o, const __fp16* s_e,
                                                            const __fp16* s_o, float epsilon, int rms_norm) {
    HVX_Vector x00_hf = *((const HVX_UVector*)in0_h);
    HVX_Vector x01_hf = *((const HVX_UVector*)(in0_h + 64));
    HVX_Vector x10_hf = *((const HVX_UVector*)in1_h);
    HVX_Vector x11_hf = *((const HVX_UVector*)(in1_h + 64));

    HVX_VectorPair x00_sf = Q6_Wsf_vcvt_Vhf(x00_hf);
    HVX_VectorPair x01_sf = Q6_Wsf_vcvt_Vhf(x01_hf);
    HVX_VectorPair x10_sf = Q6_Wsf_vcvt_Vhf(x10_hf);
    HVX_VectorPair x11_sf = Q6_Wsf_vcvt_Vhf(x11_hf);

    HVX_Vector h0_sum0 = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(x00_sf), Q6_V_lo_W(x01_sf));
    HVX_Vector h0_sum1 = Q6_Vsf_vadd_VsfVsf(Q6_V_hi_W(x00_sf), Q6_V_hi_W(x01_sf));
    HVX_Vector h1_sum0 = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(x10_sf), Q6_V_lo_W(x11_sf));
    HVX_Vector h1_sum1 = Q6_Vsf_vadd_VsfVsf(Q6_V_hi_W(x10_sf), Q6_V_hi_W(x11_sf));

    HVX_Vector h0_sq0 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(x00_sf), Q6_V_lo_W(x00_sf)),
                                           Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(x01_sf), Q6_V_lo_W(x01_sf)));
    HVX_Vector h0_sq1 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(x00_sf), Q6_V_hi_W(x00_sf)),
                                           Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(x01_sf), Q6_V_hi_W(x01_sf)));
    HVX_Vector h1_sq0 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(x10_sf), Q6_V_lo_W(x10_sf)),
                                           Q6_Vsf_vmpy_VsfVsf(Q6_V_lo_W(x11_sf), Q6_V_lo_W(x11_sf)));
    HVX_Vector h1_sq1 = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(x10_sf), Q6_V_hi_W(x10_sf)),
                                           Q6_Vsf_vmpy_VsfVsf(Q6_V_hi_W(x11_sf), Q6_V_hi_W(x11_sf)));

    float mean0 = 0.0f, mean1 = 0.0f;
    if (!rms_norm) {
        mean0 = htp_ops_rope_reduce_sum2_f32(h0_sum0, h0_sum1) / 128.0f;
        mean1 = htp_ops_rope_reduce_sum2_f32(h1_sum0, h1_sum1) / 128.0f;
    }
    float var0 = htp_ops_rope_reduce_sum2_f32(h0_sq0, h0_sq1) / 128.0f;
    float var1 = htp_ops_rope_reduce_sum2_f32(h1_sq0, h1_sq1) / 128.0f;
    if (!rms_norm) {
        var0 -= mean0 * mean0;
        var1 -= mean1 * mean1;
    }
    float inv_std0 = htp_ops_rope_fast_rsqrtf(var0 + epsilon);
    float inv_std1 = htp_ops_rope_fast_rsqrtf(var1 + epsilon);

    union { float f; int32_t i; } u_mean0 = { .f = mean0 };
    union { float f; int32_t i; } u_mean1 = { .f = mean1 };
    union { float f; int32_t i; } u_inv_std0 = { .f = inv_std0 };
    union { float f; int32_t i; } u_inv_std1 = { .f = inv_std1 };
    HVX_Vector vmean0 = Q6_V_vsplat_R(u_mean0.i);
    HVX_Vector vmean1 = Q6_V_vsplat_R(u_mean1.i);
    HVX_Vector vinv_std0 = Q6_V_vsplat_R(u_inv_std0.i);
    HVX_Vector vinv_std1 = Q6_V_vsplat_R(u_inv_std1.i);

    HVX_Vector vg00 = *((const HVX_UVector*)gamma);
    HVX_Vector vg01 = *((const HVX_UVector*)(gamma + 32));
    HVX_VectorPair g0_deal = Q6_W_vdeal_VVR(vg01, vg00, -4);
    HVX_Vector vg10 = *((const HVX_UVector*)(gamma + 64));
    HVX_Vector vg11 = *((const HVX_UVector*)(gamma + 96));
    HVX_VectorPair g1_deal = Q6_W_vdeal_VVR(vg11, vg10, -4);
    HVX_Vector vc0 = vmemu((HVX_Vector*)c_e);
    HVX_Vector vc1 = vmemu((HVX_Vector*)c_o);
    HVX_Vector vse = vmemu((HVX_Vector*)s_e);
    HVX_Vector vso = vmemu((HVX_Vector*)s_o);

    htp_ops_store_norm_rope128(out0_h, x00_hf, x01_hf, vmean0, vinv_std0,
                               Q6_V_lo_W(g0_deal), Q6_V_hi_W(g0_deal),
                               Q6_V_lo_W(g1_deal), Q6_V_hi_W(g1_deal),
                               vc0, vc1, vse, vso, rms_norm);
    htp_ops_store_norm_rope128(out1_h, x10_hf, x11_hf, vmean1, vinv_std1,
                               Q6_V_lo_W(g0_deal), Q6_V_hi_W(g0_deal),
                               Q6_V_lo_W(g1_deal), Q6_V_hi_W(g1_deal),
                               vc0, vc1, vse, vso, rms_norm);
}

static inline void compute_layernorm_rope_head128_fp16(__fp16* out_base, const __fp16* in_base,
                                                       const float* gamma, int headNumber,
                                                       const __fp16* c_e, const __fp16* c_o,
                                                       const __fp16* s_e, const __fp16* s_o,
                                                       float epsilon, int rms_norm) {
    int h = 0;
    for (; h + 2 <= headNumber; h += 2) {
        compute_layernorm_rope_two_heads128_fp16(out_base + (size_t)h * 128,
                                                 out_base + (size_t)(h + 1) * 128,
                                                 in_base + (size_t)h * 128,
                                                 in_base + (size_t)(h + 1) * 128,
                                                 gamma, c_e, c_o, s_e, s_o, epsilon, rms_norm);
    }
    for (; h < headNumber; ++h) {
        compute_layernorm_rope_one_head128_fp16(out_base + (size_t)h * 128,
                                                in_base + (size_t)h * 128,
                                                gamma, c_e, c_o, s_e, s_o, epsilon, rms_norm);
    }
}

static inline int htp_ops_rope_pick_batch_task_count(int batch_seq) {
    unsigned int worker_cap = g_max_num_workers;
    if (worker_cap <= 1 || batch_seq <= 1) {
        return 1;
    }
    return batch_seq < (int)worker_cap ? batch_seq : (int)worker_cap;
}

static inline void htp_ops_rope_fuse_layernorm_process_one_seq(
        int seq_idx,
        const __fp16* q_in, __fp16* q_out,
        const __fp16* k_in, __fp16* k_out,
        const __fp16* cos_even, const __fp16* cos_odd,
        const __fp16* sin_even, const __fp16* sin_odd,
        const float* q_gamma, const float* k_gamma,
        int num_head, int kv_num_head, int head_dim,
        int half_head_dim, int rope_half_head_dim, int rope_dim,
        float q_epsilon, float k_epsilon,
        int q_rms_norm, int k_rms_norm,
        __fp16* q_mid, __fp16* k_mid, __fp16* trig_mid, int input_c4, int seq_len) {
    const size_t q_seq_elems = (size_t)num_head * head_dim;
    const size_t k_seq_elems = (size_t)kv_num_head * head_dim;
    const __fp16* q_seq_in = q_in + (size_t)seq_idx * q_seq_elems;
    const __fp16* k_seq_in = k_in + (size_t)seq_idx * k_seq_elems;
    __fp16* q_seq_out = q_out + (size_t)seq_idx * q_seq_elems;
    __fp16* k_seq_out = k_out + (size_t)seq_idx * k_seq_elems;
    if (input_c4) {
        htp_ops_rope_unpack_c4_token(q_in, q_mid, seq_idx, seq_len, (int)q_seq_elems);
        htp_ops_rope_unpack_c4_token(k_in, k_mid, seq_idx, seq_len, (int)k_seq_elems);
        q_seq_in = q_mid;
        k_seq_in = k_mid;
    }

    const __fp16* c_e = cos_even + (size_t)seq_idx * rope_dim;
    const __fp16* c_o = cos_odd + (size_t)seq_idx * rope_dim;
    const __fp16* s_e = sin_even + (size_t)seq_idx * rope_dim;
    const __fp16* s_o = sin_odd + (size_t)seq_idx * rope_dim;
    if (q_gamma != NULL && k_gamma != NULL && head_dim == 128 &&
        half_head_dim == 64 && rope_half_head_dim == 64) {
        l2fetch(q_seq_in, VLEN, VLEN, (num_head * 128 * (int)sizeof(__fp16)) / VLEN, 0);
        l2fetch(k_seq_in, VLEN, VLEN, (kv_num_head * 128 * (int)sizeof(__fp16)) / VLEN, 0);
        if (q_rms_norm) {
            compute_rmsnorm_rope_head128_fp16(q_seq_out, q_seq_in, q_gamma, num_head,
                                              c_e, c_o, s_e, s_o, q_epsilon);
        } else {
            compute_layernorm_rope_head128_fp16(q_seq_out, q_seq_in, q_gamma, num_head,
                                                c_e, c_o, s_e, s_o, q_epsilon, q_rms_norm);
        }
        if (k_rms_norm) {
            compute_rmsnorm_rope_head128_fp16(k_seq_out, k_seq_in, k_gamma, kv_num_head,
                                              c_e, c_o, s_e, s_o, k_epsilon);
        } else {
            compute_layernorm_rope_head128_fp16(k_seq_out, k_seq_in, k_gamma, kv_num_head,
                                                c_e, c_o, s_e, s_o, k_epsilon, k_rms_norm);
        }
        return;
    }
    memcpy(trig_mid, c_e, (size_t)half_head_dim * sizeof(__fp16));
    memcpy(trig_mid + half_head_dim, c_o, (size_t)half_head_dim * sizeof(__fp16));
    memcpy(trig_mid + head_dim, s_e, (size_t)half_head_dim * sizeof(__fp16));
    memcpy(trig_mid + head_dim + half_head_dim, s_o, (size_t)half_head_dim * sizeof(__fp16));
    c_e = trig_mid;
    c_o = trig_mid + half_head_dim;
    s_e = trig_mid + head_dim;
    s_o = trig_mid + head_dim + half_head_dim;

    const __fp16* q_rope_in = q_seq_in;
    if (q_gamma != NULL) {
        compute_layernorm_head_fp16(q_mid, q_seq_in, q_gamma, num_head, head_dim, q_epsilon, q_rms_norm);
        q_rope_in = q_mid;
    }
    compute_rope_head_fp16(q_seq_out, q_rope_in, num_head, head_dim, c_e, c_o, s_e, s_o,
                           rope_half_head_dim, half_head_dim);

    const __fp16* k_rope_in = k_seq_in;
    if (k_gamma != NULL) {
        compute_layernorm_head_fp16(k_mid, k_seq_in, k_gamma, kv_num_head, head_dim, k_epsilon, k_rms_norm);
        k_rope_in = k_mid;
    }
    compute_rope_head_fp16(k_seq_out, k_rope_in, kv_num_head, head_dim, c_e, c_o, s_e, s_o,
                           rope_half_head_dim, half_head_dim);
}

static void htp_ops_rope_fuse_layernorm_batch_worker(void* data, int worker_index) {
    HtpOpsRopeFuseLayernormBatchTaskState* state = (HtpOpsRopeFuseLayernormBatchTaskState*)data;
    uint8_t* worker_vtcm = state->workspace_base + (size_t)worker_index * state->worker_vtcm_bytes;
    uint8_t* worker_vtcm_ptr = worker_vtcm;
    __fp16* q_mid = (__fp16*)vtcm_seq_alloc(&worker_vtcm_ptr, (size_t)state->num_head * state->head_dim * sizeof(__fp16));
    __fp16* k_mid = (__fp16*)vtcm_seq_alloc(&worker_vtcm_ptr, (size_t)state->kv_num_head * state->head_dim * sizeof(__fp16));
    __fp16* trig_mid = (__fp16*)vtcm_seq_alloc(&worker_vtcm_ptr, (size_t)2 * state->head_dim * sizeof(__fp16));

    while (1) {
        unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
        if ((int)task_id >= state->batch_seq) {
            break;
        }
        htp_ops_rope_fuse_layernorm_process_one_seq((int)task_id,
                                                    state->q_in, state->q_out,
                                                    state->k_in, state->k_out,
                                                    state->cos_even, state->cos_odd,
                                                    state->sin_even, state->sin_odd,
                                                    state->q_gamma, state->k_gamma,
                                                    state->num_head, state->kv_num_head, state->head_dim,
                                                    state->half_head_dim, state->rope_half_head_dim, state->rope_dim,
                                                    state->q_epsilon, state->k_epsilon,
                                                    state->q_rms_norm, state->k_rms_norm,
                                                    q_mid, k_mid, trig_mid,
                                                    state->input_c4, state->batch_seq);
    }

    worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

static void htp_ops_rope_batch_worker(void* data, int worker_index) {
    (void)worker_index;
    HtpOpsRopeBatchTaskState* state = (HtpOpsRopeBatchTaskState*)data;
    const size_t q_seq_elems = (size_t)state->num_head * state->head_dim;
    const size_t k_seq_elems = (size_t)state->kv_num_head * state->head_dim;

    while (1) {
        unsigned int task_id = worker_pool_atomic_inc_return(&(state->task_id)) - 1;
        if ((int)task_id >= state->batch_seq) {
            break;
        }
        const int i = (int)task_id;
        const __fp16* c_e = state->cos_even + (size_t)i * state->rope_dim;
        const __fp16* c_o = state->cos_odd + (size_t)i * state->rope_dim;
        const __fp16* s_e = state->sin_even + (size_t)i * state->rope_dim;
        const __fp16* s_o = state->sin_odd + (size_t)i * state->rope_dim;
        if (state->input_c4) {
            compute_rope_head64blocks_c4_fp16(state->q_out + (size_t)i * q_seq_elems,
                                              state->q_in, i, state->batch_seq, state->num_head,
                                              state->head_dim, state->half_head_dim,
                                              c_e, c_o, s_e, s_o, state->rope_half_head_dim);
            compute_rope_head64blocks_c4_fp16(state->k_out + (size_t)i * k_seq_elems,
                                              state->k_in, i, state->batch_seq, state->kv_num_head,
                                              state->head_dim, state->half_head_dim,
                                              c_e, c_o, s_e, s_o, state->rope_half_head_dim);
            continue;
        }
        const __fp16* q_seq_in = state->q_in + (size_t)i * q_seq_elems;
        const __fp16* k_seq_in = state->k_in + (size_t)i * k_seq_elems;
        compute_rope_head_fp16(state->q_out + (size_t)i * q_seq_elems,
                               q_seq_in,
                               state->num_head, state->head_dim, c_e, c_o, s_e, s_o,
                               state->rope_half_head_dim, state->half_head_dim);
        compute_rope_head_fp16(state->k_out + (size_t)i * k_seq_elems,
                               k_seq_in,
                               state->kv_num_head, state->head_dim, c_e, c_o, s_e, s_o,
                               state->rope_half_head_dim, state->half_head_dim);
    }

    worker_pool_synctoken_jobdone(&(state->sync_ctx));
}

// Helper function to apply RoPE vector logic for multiple heads in a batch.
static inline void compute_rope_head_fp16(__fp16* out_base, const __fp16* in_base, int num_head, int head_dim,
                                          const __fp16* c_e, const __fp16* c_o,
                                          const __fp16* s_e, const __fp16* s_o,
                                          int rope_half_head_dim, int half_head_dim) {
    if (head_dim >= 128 && (half_head_dim % 64) == 0 &&
        rope_half_head_dim > 0 && rope_half_head_dim <= half_head_dim) {
        compute_rope_head64blocks_fp16(out_base, in_base, num_head, head_dim, half_head_dim,
                                       c_e, c_o, s_e, s_o, rope_half_head_dim);
        return;
    }
    int d = 0;
    int pack = 64;
#ifdef __HVX_LENGTH__
    pack = __HVX_LENGTH__ / sizeof(__fp16);
#endif

    if (head_dim < pack) {
        for (int h = 0; h < num_head; ++h) {
            const __fp16* in_h = in_base + (size_t)h * head_dim;
            __fp16* out_h = out_base + (size_t)h * head_dim;
            int x = 0;
            for (; x < rope_half_head_dim; ++x) {
                const float q0 = (float)in_h[x];
                const float q1 = (float)in_h[x + half_head_dim];
                out_h[x] = (__fp16)(q0 * (float)c_e[x] - q1 * (float)s_e[x]);
                out_h[x + half_head_dim] = (__fp16)(q1 * (float)c_o[x] + q0 * (float)s_o[x]);
            }
            for (; x < half_head_dim; ++x) {
                out_h[x] = in_h[x];
                out_h[x + half_head_dim] = in_h[x + half_head_dim];
            }
        }
        return;
    }

    for (; d + pack <= rope_half_head_dim; d += pack) {
        HVX_Vector v_c0 = vmemu((HVX_Vector*)&c_e[d]);
        HVX_Vector v_c1 = vmemu((HVX_Vector*)&c_o[d]);
        HVX_Vector v_se = vmemu((HVX_Vector*)&s_e[d]);
        HVX_Vector v_so = vmemu((HVX_Vector*)&s_o[d]);

        int h = 0;
        for (; h + 2 <= num_head; h += 2) {
            const __fp16* in_h0 = in_base + (size_t)h * head_dim;
            const __fp16* in_h1 = in_h0 + head_dim;
            __fp16* out_h0 = out_base + (size_t)h * head_dim;
            __fp16* out_h1 = out_h0 + head_dim;

            HVX_Vector v_q00 = vmemu((HVX_Vector*)&in_h0[d]);
            HVX_Vector v_q01 = vmemu((HVX_Vector*)&in_h0[d + half_head_dim]);
            HVX_Vector v_q10 = vmemu((HVX_Vector*)&in_h1[d]);
            HVX_Vector v_q11 = vmemu((HVX_Vector*)&in_h1[d + half_head_dim]);

            HVX_Vector v_q00c0 = Q6_Vqf16_vmpy_VhfVhf(v_q00, v_c0);
            HVX_Vector v_q01se = Q6_Vqf16_vmpy_VhfVhf(v_q01, v_se);
            HVX_Vector v_res00 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_Vqf16Vqf16(v_q00c0, v_q01se));
            vmemu((HVX_Vector*)&out_h0[d]) = v_res00;

            HVX_Vector v_q01c1 = Q6_Vqf16_vmpy_VhfVhf(v_q01, v_c1);
            HVX_Vector v_q00so = Q6_Vqf16_vmpy_VhfVhf(v_q00, v_so);
            HVX_Vector v_res01 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(v_q01c1, v_q00so));
            vmemu((HVX_Vector*)&out_h0[d + half_head_dim]) = v_res01;

            HVX_Vector v_q10c0 = Q6_Vqf16_vmpy_VhfVhf(v_q10, v_c0);
            HVX_Vector v_q11se = Q6_Vqf16_vmpy_VhfVhf(v_q11, v_se);
            HVX_Vector v_res10 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_Vqf16Vqf16(v_q10c0, v_q11se));
            vmemu((HVX_Vector*)&out_h1[d]) = v_res10;

            HVX_Vector v_q11c1 = Q6_Vqf16_vmpy_VhfVhf(v_q11, v_c1);
            HVX_Vector v_q10so = Q6_Vqf16_vmpy_VhfVhf(v_q10, v_so);
            HVX_Vector v_res11 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(v_q11c1, v_q10so));
            vmemu((HVX_Vector*)&out_h1[d + half_head_dim]) = v_res11;
        }

        for (; h < num_head; ++h) {
            const __fp16* in_h = in_base + (size_t)h * head_dim;
            __fp16* out_h = out_base + (size_t)h * head_dim;
            HVX_Vector v_q0 = vmemu((HVX_Vector*)&in_h[d]);
            HVX_Vector v_q1 = vmemu((HVX_Vector*)&in_h[d + half_head_dim]);

            HVX_Vector v_q0c0 = Q6_Vqf16_vmpy_VhfVhf(v_q0, v_c0);
            HVX_Vector v_q1se = Q6_Vqf16_vmpy_VhfVhf(v_q1, v_se);
            HVX_Vector v_res0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_Vqf16Vqf16(v_q0c0, v_q1se));
            vmemu((HVX_Vector*)&out_h[d]) = v_res0;

            HVX_Vector v_q1c1 = Q6_Vqf16_vmpy_VhfVhf(v_q1, v_c1);
            HVX_Vector v_q0so = Q6_Vqf16_vmpy_VhfVhf(v_q0, v_so);
            HVX_Vector v_res1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(v_q1c1, v_q0so));
            vmemu((HVX_Vector*)&out_h[d + half_head_dim]) = v_res1;
        }
    }

    int remain_rope = rope_half_head_dim - d;
    if (remain_rope > 0) {
        __fp16 tmp_c[128 / sizeof(__fp16)] __attribute__((aligned(128))) = {0};
        __fp16 tmp_co[128 / sizeof(__fp16)] __attribute__((aligned(128))) = {0};
        __fp16 tmp_s[128 / sizeof(__fp16)] __attribute__((aligned(128))) = {0};
        __fp16 tmp_so[128 / sizeof(__fp16)] __attribute__((aligned(128))) = {0};
        memcpy(tmp_c, &c_e[d], remain_rope * sizeof(__fp16));
        memcpy(tmp_co, &c_o[d], remain_rope * sizeof(__fp16));
        memcpy(tmp_s, &s_e[d], remain_rope * sizeof(__fp16));
        memcpy(tmp_so, &s_o[d], remain_rope * sizeof(__fp16));
        HVX_Vector v_c0 = vmem((HVX_Vector*)tmp_c);
        HVX_Vector v_c1 = vmem((HVX_Vector*)tmp_co);
        HVX_Vector v_se = vmem((HVX_Vector*)tmp_s);
        HVX_Vector v_so = vmem((HVX_Vector*)tmp_so);

        int h = 0;
        for (; h + 2 <= num_head; h += 2) {
            const __fp16* in_h0 = in_base + (size_t)h * head_dim;
            const __fp16* in_h1 = in_h0 + head_dim;
            __fp16* out_h0 = out_base + (size_t)h * head_dim;
            __fp16* out_h1 = out_h0 + head_dim;

            __fp16 tmp_in00[128 / sizeof(__fp16)] __attribute__((aligned(128))) = {0};
            __fp16 tmp_in01[128 / sizeof(__fp16)] __attribute__((aligned(128))) = {0};
            __fp16 tmp_in10[128 / sizeof(__fp16)] __attribute__((aligned(128))) = {0};
            __fp16 tmp_in11[128 / sizeof(__fp16)] __attribute__((aligned(128))) = {0};
            memcpy(tmp_in00, &in_h0[d], remain_rope * sizeof(__fp16));
            memcpy(tmp_in01, &in_h0[d + half_head_dim], remain_rope * sizeof(__fp16));
            memcpy(tmp_in10, &in_h1[d], remain_rope * sizeof(__fp16));
            memcpy(tmp_in11, &in_h1[d + half_head_dim], remain_rope * sizeof(__fp16));

            HVX_Vector v_q00 = vmem((HVX_Vector*)tmp_in00);
            HVX_Vector v_q01 = vmem((HVX_Vector*)tmp_in01);
            HVX_Vector v_q10 = vmem((HVX_Vector*)tmp_in10);
            HVX_Vector v_q11 = vmem((HVX_Vector*)tmp_in11);

            HVX_Vector v_q00c0 = Q6_Vqf16_vmpy_VhfVhf(v_q00, v_c0);
            HVX_Vector v_q01se = Q6_Vqf16_vmpy_VhfVhf(v_q01, v_se);
            HVX_Vector v_res00 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_Vqf16Vqf16(v_q00c0, v_q01se));

            HVX_Vector v_q01c1 = Q6_Vqf16_vmpy_VhfVhf(v_q01, v_c1);
            HVX_Vector v_q00so = Q6_Vqf16_vmpy_VhfVhf(v_q00, v_so);
            HVX_Vector v_res01 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(v_q01c1, v_q00so));

            HVX_Vector v_q10c0 = Q6_Vqf16_vmpy_VhfVhf(v_q10, v_c0);
            HVX_Vector v_q11se = Q6_Vqf16_vmpy_VhfVhf(v_q11, v_se);
            HVX_Vector v_res10 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_Vqf16Vqf16(v_q10c0, v_q11se));

            HVX_Vector v_q11c1 = Q6_Vqf16_vmpy_VhfVhf(v_q11, v_c1);
            HVX_Vector v_q10so = Q6_Vqf16_vmpy_VhfVhf(v_q10, v_so);
            HVX_Vector v_res11 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(v_q11c1, v_q10so));

            __fp16 tmp_out00[128 / sizeof(__fp16)] __attribute__((aligned(128)));
            __fp16 tmp_out01[128 / sizeof(__fp16)] __attribute__((aligned(128)));
            __fp16 tmp_out10[128 / sizeof(__fp16)] __attribute__((aligned(128)));
            __fp16 tmp_out11[128 / sizeof(__fp16)] __attribute__((aligned(128)));

            vmem((HVX_Vector*)tmp_out00) = v_res00;
            vmem((HVX_Vector*)tmp_out01) = v_res01;
            vmem((HVX_Vector*)tmp_out10) = v_res10;
            vmem((HVX_Vector*)tmp_out11) = v_res11;

            memcpy(&out_h0[d], tmp_out00, remain_rope * sizeof(__fp16));
            memcpy(&out_h0[d + half_head_dim], tmp_out01, remain_rope * sizeof(__fp16));
            memcpy(&out_h1[d], tmp_out10, remain_rope * sizeof(__fp16));
            memcpy(&out_h1[d + half_head_dim], tmp_out11, remain_rope * sizeof(__fp16));
        }

        for (; h < num_head; ++h) {
            const __fp16* in_h = in_base + (size_t)h * head_dim;
            __fp16* out_h = out_base + (size_t)h * head_dim;

            __fp16 tmp_in0[128 / sizeof(__fp16)] __attribute__((aligned(128))) = {0};
            __fp16 tmp_in1[128 / sizeof(__fp16)] __attribute__((aligned(128))) = {0};
            memcpy(tmp_in0, &in_h[d], remain_rope * sizeof(__fp16));
            memcpy(tmp_in1, &in_h[d + half_head_dim], remain_rope * sizeof(__fp16));

            HVX_Vector v_q0 = vmem((HVX_Vector*)tmp_in0);
            HVX_Vector v_q1 = vmem((HVX_Vector*)tmp_in1);

            HVX_Vector v_q0c0 = Q6_Vqf16_vmpy_VhfVhf(v_q0, v_c0);
            HVX_Vector v_q1se = Q6_Vqf16_vmpy_VhfVhf(v_q1, v_se);
            HVX_Vector v_res0 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_Vqf16Vqf16(v_q0c0, v_q1se));

            HVX_Vector v_q1c1 = Q6_Vqf16_vmpy_VhfVhf(v_q1, v_c1);
            HVX_Vector v_q0so = Q6_Vqf16_vmpy_VhfVhf(v_q0, v_so);
            HVX_Vector v_res1 = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(v_q1c1, v_q0so));

            __fp16 tmp_out0[128 / sizeof(__fp16)] __attribute__((aligned(128)));
            __fp16 tmp_out1[128 / sizeof(__fp16)] __attribute__((aligned(128)));
            vmem((HVX_Vector*)tmp_out0) = v_res0;
            vmem((HVX_Vector*)tmp_out1) = v_res1;

            memcpy(&out_h[d], tmp_out0, remain_rope * sizeof(__fp16));
            memcpy(&out_h[d + half_head_dim], tmp_out1, remain_rope * sizeof(__fp16));
        }
        d += remain_rope;
    }

    for (; d + pack <= half_head_dim; d += pack) {
        int h = 0;
        for (; h + 2 <= num_head; h += 2) {
            const __fp16* in_h0 = in_base + (size_t)h * head_dim;
            const __fp16* in_h1 = in_h0 + head_dim;
            __fp16* out_h0 = out_base + (size_t)h * head_dim;
            __fp16* out_h1 = out_h0 + head_dim;
            vmemu((HVX_Vector*)&out_h0[d]) = vmemu((HVX_Vector*)&in_h0[d]);
            vmemu((HVX_Vector*)&out_h0[d + half_head_dim]) = vmemu((HVX_Vector*)&in_h0[d + half_head_dim]);
            vmemu((HVX_Vector*)&out_h1[d]) = vmemu((HVX_Vector*)&in_h1[d]);
            vmemu((HVX_Vector*)&out_h1[d + half_head_dim]) = vmemu((HVX_Vector*)&in_h1[d + half_head_dim]);
        }
        for (; h < num_head; ++h) {
            const __fp16* in_h = in_base + (size_t)h * head_dim;
            __fp16* out_h = out_base + (size_t)h * head_dim;
            vmemu((HVX_Vector*)&out_h[d]) = vmemu((HVX_Vector*)&in_h[d]);
            vmemu((HVX_Vector*)&out_h[d + half_head_dim]) = vmemu((HVX_Vector*)&in_h[d + half_head_dim]);
        }
    }

    int remain_half = half_head_dim - d;
    if (remain_half > 0) {
        int h = 0;
        for (; h + 2 <= num_head; h += 2) {
            const __fp16* in_h0 = in_base + (size_t)h * head_dim;
            const __fp16* in_h1 = in_h0 + head_dim;
            __fp16* out_h0 = out_base + (size_t)h * head_dim;
            __fp16* out_h1 = out_h0 + head_dim;

            memcpy(&out_h0[d], &in_h0[d], remain_half * sizeof(__fp16));
            memcpy(&out_h0[d + half_head_dim], &in_h0[d + half_head_dim], remain_half * sizeof(__fp16));
            memcpy(&out_h1[d], &in_h1[d], remain_half * sizeof(__fp16));
            memcpy(&out_h1[d + half_head_dim], &in_h1[d + half_head_dim], remain_half * sizeof(__fp16));
        }
        for (; h < num_head; ++h) {
            const __fp16* in_h = in_base + (size_t)h * head_dim;
            __fp16* out_h = out_base + (size_t)h * head_dim;

            memcpy(&out_h[d], &in_h[d], remain_half * sizeof(__fp16));
            memcpy(&out_h[d + half_head_dim], &in_h[d + half_head_dim], remain_half * sizeof(__fp16));
        }
        d += remain_half;
    }
}

// NCHW layout RoPE computation
AEEResult htp_ops_rope(uint8_t* q_out_ptr, uint8_t* q_in_ptr,
                           uint8_t* k_out_ptr, uint8_t* k_in_ptr,
                           uint8_t* cos_even_ptr, uint8_t* cos_odd_ptr,
                           uint8_t* sin_even_ptr, uint8_t* sin_odd_ptr,
                           int32_t batch_seq, int32_t num_head, int32_t kv_num_head,
                           int32_t head_dim, int32_t rope_dim, int32_t input_c4) {

    if (head_dim % 2 != 0) return -1;

    int half_head_dim = head_dim / 2;
    int rope_half_head_dim = rope_dim / 2;
    if (rope_half_head_dim > half_head_dim) rope_half_head_dim = half_head_dim;

    __fp16* q_out = (__fp16*)q_out_ptr;
    const __fp16* q_in = (const __fp16*)q_in_ptr;
    __fp16* k_out = (__fp16*)k_out_ptr;
    const __fp16* k_in = (const __fp16*)k_in_ptr;
    const __fp16* cos_even = (const __fp16*)cos_even_ptr;
    const __fp16* cos_odd = (const __fp16*)cos_odd_ptr;
    const __fp16* sin_even = (const __fp16*)sin_even_ptr;
    const __fp16* sin_odd = (const __fp16*)sin_odd_ptr;

    size_t q_size = (size_t)batch_seq * num_head * head_dim * sizeof(__fp16); (void)q_size;
    size_t k_size = (size_t)batch_seq * kv_num_head * head_dim * sizeof(__fp16); (void)k_size;
    size_t trig_size = (size_t)batch_seq * rope_dim * sizeof(__fp16); (void)trig_size;


    const int n_tasks = htp_ops_rope_pick_batch_task_count(batch_seq);
    const size_t q_seq_elems = (size_t)num_head * head_dim;
    const size_t k_seq_elems = (size_t)kv_num_head * head_dim;
    if (n_tasks <= 1) {
        for (int i = 0; i < batch_seq; ++i) {
            const __fp16* c_e = cos_even + (size_t)i * rope_dim;
            const __fp16* c_o = cos_odd + (size_t)i * rope_dim;
            const __fp16* s_e = sin_even + (size_t)i * rope_dim;
            const __fp16* s_o = sin_odd + (size_t)i * rope_dim;
            if (input_c4) {
                compute_rope_head64blocks_c4_fp16(q_out + (size_t)i * q_seq_elems,
                                                  q_in, i, batch_seq, num_head,
                                                  head_dim, half_head_dim,
                                                  c_e, c_o, s_e, s_o, rope_half_head_dim);
                compute_rope_head64blocks_c4_fp16(k_out + (size_t)i * k_seq_elems,
                                                  k_in, i, batch_seq, kv_num_head,
                                                  head_dim, half_head_dim,
                                                  c_e, c_o, s_e, s_o, rope_half_head_dim);
                continue;
            }
            const __fp16* q_seq_in = q_in + (size_t)i * q_seq_elems;
            const __fp16* k_seq_in = k_in + (size_t)i * k_seq_elems;
            compute_rope_head_fp16(q_out + (size_t)i * q_seq_elems,
                                   q_seq_in,
                                   num_head, head_dim, c_e, c_o, s_e, s_o,
                                   rope_half_head_dim, half_head_dim);

            compute_rope_head_fp16(k_out + (size_t)i * k_seq_elems,
                                   k_seq_in,
                                   kv_num_head, head_dim, c_e, c_o, s_e, s_o,
                                   rope_half_head_dim, half_head_dim);
        }
    } else {
        HtpOpsRopeBatchTaskState state = {};
        state.batch_seq = batch_seq;
        state.num_head = num_head;
        state.kv_num_head = kv_num_head;
        state.head_dim = head_dim;
        state.half_head_dim = half_head_dim;
        state.rope_half_head_dim = rope_half_head_dim;
        state.rope_dim = rope_dim;
        state.q_in = q_in;
        state.q_out = q_out;
        state.k_in = k_in;
        state.k_out = k_out;
        state.cos_even = cos_even;
        state.cos_odd = cos_odd;
        state.sin_even = sin_even;
        state.sin_odd = sin_odd;
        state.input_c4 = input_c4;

        worker_pool_job_t job;
        job.fptr = htp_ops_rope_batch_worker;
        job.dptr = &state;
        worker_pool_synctoken_init(&(state.sync_ctx), n_tasks);
        for (int i = 0; i < n_tasks; ++i) {
            worker_pool_submit(NULL, job);
        }
        worker_pool_synctoken_wait(&(state.sync_ctx));
    }

    return AEE_SUCCESS;
}

static inline void compute_layernorm_one_head_fp16(__fp16* out_h, const __fp16* in_h, const float* gamma,
                                                   int head_dim, float epsilon, int rms_norm) {
    __fp16* __restrict out_ptr = out_h;
    const __fp16* __restrict in_ptr = in_h;
    const float* __restrict gamma_ptr = gamma;

    float sum = 0.0f;
    float sqsum = 0.0f;
    const int fullUnit = head_dim >> 6;
    const int vec_end = fullUnit << 6;

    HVX_Vector vsqsum0 = Q6_V_vsplat_R(0);
    HVX_Vector vsqsum1 = Q6_V_vsplat_R(0);
    HVX_Vector vsum0 = Q6_V_vsplat_R(0);
    HVX_Vector vsum1 = Q6_V_vsplat_R(0);

    if (rms_norm) {
        const __fp16* vec_in_ptr = in_ptr;
        for (int i = 0; i < fullUnit; ++i, vec_in_ptr += 64) {
            HVX_Vector s_hf = *((const HVX_UVector*)vec_in_ptr);
            HVX_VectorPair s_sf = Q6_Wsf_vcvt_Vhf(s_hf);
            HVX_Vector s_sf0 = Q6_V_lo_W(s_sf);
            HVX_Vector s_sf1 = Q6_V_hi_W(s_sf);

            vsqsum0 = Q6_Vsf_vadd_VsfVsf(vsqsum0, Q6_Vsf_vmpy_VsfVsf(s_sf0, s_sf0));
            vsqsum1 = Q6_Vsf_vadd_VsfVsf(vsqsum1, Q6_Vsf_vmpy_VsfVsf(s_sf1, s_sf1));
        }
    } else {
        const __fp16* vec_in_ptr = in_ptr;
        for (int i = 0; i < fullUnit; ++i, vec_in_ptr += 64) {
            HVX_Vector s_hf = *((const HVX_UVector*)vec_in_ptr);
            HVX_VectorPair s_sf = Q6_Wsf_vcvt_Vhf(s_hf);
            HVX_Vector s_sf0 = Q6_V_lo_W(s_sf);
            HVX_Vector s_sf1 = Q6_V_hi_W(s_sf);

            vsum0 = Q6_Vsf_vadd_VsfVsf(vsum0, s_sf0);
            vsum1 = Q6_Vsf_vadd_VsfVsf(vsum1, s_sf1);
            vsqsum0 = Q6_Vsf_vadd_VsfVsf(vsqsum0, Q6_Vsf_vmpy_VsfVsf(s_sf0, s_sf0));
            vsqsum1 = Q6_Vsf_vadd_VsfVsf(vsqsum1, Q6_Vsf_vmpy_VsfVsf(s_sf1, s_sf1));
        }
    }

    if (fullUnit > 0) {
        HVX_Vector vsqsum = Q6_Vsf_vadd_VsfVsf(vsqsum0, vsqsum1);
        vsqsum = Q6_Vsf_vadd_VsfVsf(vsqsum, Q6_V_vror_VR(vsqsum, 64));
        vsqsum = Q6_Vsf_vadd_VsfVsf(vsqsum, Q6_V_vror_VR(vsqsum, 32));
        vsqsum = Q6_Vsf_vadd_VsfVsf(vsqsum, Q6_V_vror_VR(vsqsum, 16));
        vsqsum = Q6_Vsf_vadd_VsfVsf(vsqsum, Q6_V_vror_VR(vsqsum, 8));
        vsqsum = Q6_Vsf_vadd_VsfVsf(vsqsum, Q6_V_vror_VR(vsqsum, 4));
        union { HVX_Vector v; float f[32]; } usqsum = { .v = vsqsum };
        sqsum = usqsum.f[0];

        if (!rms_norm) {
            HVX_Vector vsum = Q6_Vsf_vadd_VsfVsf(vsum0, vsum1);
            vsum = Q6_Vsf_vadd_VsfVsf(vsum, Q6_V_vror_VR(vsum, 64));
            vsum = Q6_Vsf_vadd_VsfVsf(vsum, Q6_V_vror_VR(vsum, 32));
            vsum = Q6_Vsf_vadd_VsfVsf(vsum, Q6_V_vror_VR(vsum, 16));
            vsum = Q6_Vsf_vadd_VsfVsf(vsum, Q6_V_vror_VR(vsum, 8));
            vsum = Q6_Vsf_vadd_VsfVsf(vsum, Q6_V_vror_VR(vsum, 4));
            union { HVX_Vector v; float f[32]; } usum = { .v = vsum };
            sum = usum.f[0];
        }
    }

    for (int i = vec_end; i < head_dim; ++i) {
        float val = (float)in_ptr[i];
        if (!rms_norm) {
            sum += val;
        }
        sqsum += val * val;
    }

    float mean = 0.0f;
    if (!rms_norm) {
        mean = sum / head_dim;
    }
    float variance = sqsum / head_dim;
    if (!rms_norm) {
        variance -= mean * mean;
    }
    float inv_std = 1.0f / __builtin_sqrtf(variance + epsilon);

    union { float f; int32_t i; } u_mean = { .f = mean };
    union { float f; int32_t i; } u_inv_std = { .f = inv_std };
    HVX_Vector vmean = Q6_V_vsplat_R(u_mean.i);
    HVX_Vector vinv_std = Q6_V_vsplat_R(u_inv_std.i);

    const __fp16* vec_in_ptr = in_ptr;
    __fp16* vec_out_ptr = out_ptr;
    const float* vec_gamma_ptr = gamma_ptr;
    for (int i = 0; i < fullUnit; ++i, vec_in_ptr += 64, vec_out_ptr += 64, vec_gamma_ptr += 64) {
        HVX_Vector s_hf = *((const HVX_UVector*)vec_in_ptr);
        HVX_VectorPair s_sf = Q6_Wsf_vcvt_Vhf(s_hf);
        HVX_Vector s_sf0 = Q6_V_lo_W(s_sf);
        HVX_Vector s_sf1 = Q6_V_hi_W(s_sf);

        HVX_Vector out0 = rms_norm ? s_sf0 : Q6_Vsf_vsub_VsfVsf(s_sf0, vmean);
        HVX_Vector out1 = rms_norm ? s_sf1 : Q6_Vsf_vsub_VsfVsf(s_sf1, vmean);
        out0 = Q6_Vsf_vmpy_VsfVsf(out0, vinv_std);
        out1 = Q6_Vsf_vmpy_VsfVsf(out1, vinv_std);

        HVX_Vector vg0 = *((const HVX_UVector*)vec_gamma_ptr);
        HVX_Vector vg1 = *((const HVX_UVector*)(vec_gamma_ptr + 32));
        HVX_VectorPair g_deal = Q6_W_vdeal_VVR(vg1, vg0, -4);
        out0 = Q6_Vsf_vmpy_VsfVsf(out0, Q6_V_lo_W(g_deal));
        out1 = Q6_Vsf_vmpy_VsfVsf(out1, Q6_V_hi_W(g_deal));

        HVX_Vector out_hf = Q6_Vhf_vcvt_VsfVsf(out0, out1);
        *((HVX_UVector*)vec_out_ptr) = out_hf;
    }

    for (int i = vec_end; i < head_dim; ++i) {
        float val = (float)in_ptr[i];
        float norm_val = (val - mean) * inv_std;
        norm_val *= gamma_ptr[i];
        out_ptr[i] = (__fp16)norm_val;
    }
}

static inline void compute_layernorm_two_heads_fp16(__fp16* out0_h, __fp16* out1_h,
                                                    const __fp16* in0_h, const __fp16* in1_h,
                                                    const float* gamma, int head_dim,
                                                    float epsilon, int rms_norm) {
    __fp16* __restrict out0_ptr = out0_h;
    __fp16* __restrict out1_ptr = out1_h;
    const __fp16* __restrict in0_ptr = in0_h;
    const __fp16* __restrict in1_ptr = in1_h;
    const float* __restrict gamma_ptr = gamma;

    float sum0 = 0.0f, sum1 = 0.0f;
    float sqsum0 = 0.0f, sqsum1 = 0.0f;
    const int fullUnit = head_dim >> 6;
    const int vec_end = fullUnit << 6;

    HVX_Vector v0_sqsum0 = Q6_V_vsplat_R(0);
    HVX_Vector v0_sqsum1 = Q6_V_vsplat_R(0);
    HVX_Vector v1_sqsum0 = Q6_V_vsplat_R(0);
    HVX_Vector v1_sqsum1 = Q6_V_vsplat_R(0);
    HVX_Vector v0_sum0 = Q6_V_vsplat_R(0);
    HVX_Vector v0_sum1 = Q6_V_vsplat_R(0);
    HVX_Vector v1_sum0 = Q6_V_vsplat_R(0);
    HVX_Vector v1_sum1 = Q6_V_vsplat_R(0);

    const __fp16* vec_in0_ptr = in0_ptr;
    const __fp16* vec_in1_ptr = in1_ptr;
    if (rms_norm) {
        for (int i = 0; i < fullUnit; ++i, vec_in0_ptr += 64, vec_in1_ptr += 64) {
            HVX_Vector s0_hf = *((const HVX_UVector*)vec_in0_ptr);
            HVX_Vector s1_hf = *((const HVX_UVector*)vec_in1_ptr);

            HVX_VectorPair s0_sf = Q6_Wsf_vcvt_Vhf(s0_hf);
            HVX_VectorPair s1_sf = Q6_Wsf_vcvt_Vhf(s1_hf);

            HVX_Vector s00 = Q6_V_lo_W(s0_sf);
            HVX_Vector s01 = Q6_V_hi_W(s0_sf);
            HVX_Vector s10 = Q6_V_lo_W(s1_sf);
            HVX_Vector s11 = Q6_V_hi_W(s1_sf);

            v0_sqsum0 = Q6_Vsf_vadd_VsfVsf(v0_sqsum0, Q6_Vsf_vmpy_VsfVsf(s00, s00));
            v0_sqsum1 = Q6_Vsf_vadd_VsfVsf(v0_sqsum1, Q6_Vsf_vmpy_VsfVsf(s01, s01));
            v1_sqsum0 = Q6_Vsf_vadd_VsfVsf(v1_sqsum0, Q6_Vsf_vmpy_VsfVsf(s10, s10));
            v1_sqsum1 = Q6_Vsf_vadd_VsfVsf(v1_sqsum1, Q6_Vsf_vmpy_VsfVsf(s11, s11));
        }
    } else {
        for (int i = 0; i < fullUnit; ++i, vec_in0_ptr += 64, vec_in1_ptr += 64) {
            HVX_Vector s0_hf = *((const HVX_UVector*)vec_in0_ptr);
            HVX_Vector s1_hf = *((const HVX_UVector*)vec_in1_ptr);

            HVX_VectorPair s0_sf = Q6_Wsf_vcvt_Vhf(s0_hf);
            HVX_VectorPair s1_sf = Q6_Wsf_vcvt_Vhf(s1_hf);

            HVX_Vector s00 = Q6_V_lo_W(s0_sf);
            HVX_Vector s01 = Q6_V_hi_W(s0_sf);
            HVX_Vector s10 = Q6_V_lo_W(s1_sf);
            HVX_Vector s11 = Q6_V_hi_W(s1_sf);

            v0_sum0 = Q6_Vsf_vadd_VsfVsf(v0_sum0, s00);
            v0_sum1 = Q6_Vsf_vadd_VsfVsf(v0_sum1, s01);
            v1_sum0 = Q6_Vsf_vadd_VsfVsf(v1_sum0, s10);
            v1_sum1 = Q6_Vsf_vadd_VsfVsf(v1_sum1, s11);
            v0_sqsum0 = Q6_Vsf_vadd_VsfVsf(v0_sqsum0, Q6_Vsf_vmpy_VsfVsf(s00, s00));
            v0_sqsum1 = Q6_Vsf_vadd_VsfVsf(v0_sqsum1, Q6_Vsf_vmpy_VsfVsf(s01, s01));
            v1_sqsum0 = Q6_Vsf_vadd_VsfVsf(v1_sqsum0, Q6_Vsf_vmpy_VsfVsf(s10, s10));
            v1_sqsum1 = Q6_Vsf_vadd_VsfVsf(v1_sqsum1, Q6_Vsf_vmpy_VsfVsf(s11, s11));
        }
    }

    if (fullUnit > 0) {
        HVX_Vector vsqsum0_v = Q6_Vsf_vadd_VsfVsf(v0_sqsum0, v0_sqsum1);
        HVX_Vector vsqsum1_v = Q6_Vsf_vadd_VsfVsf(v1_sqsum0, v1_sqsum1);
        vsqsum0_v = Q6_Vsf_vadd_VsfVsf(vsqsum0_v, Q6_V_vror_VR(vsqsum0_v, 64));
        vsqsum1_v = Q6_Vsf_vadd_VsfVsf(vsqsum1_v, Q6_V_vror_VR(vsqsum1_v, 64));
        vsqsum0_v = Q6_Vsf_vadd_VsfVsf(vsqsum0_v, Q6_V_vror_VR(vsqsum0_v, 32));
        vsqsum1_v = Q6_Vsf_vadd_VsfVsf(vsqsum1_v, Q6_V_vror_VR(vsqsum1_v, 32));
        vsqsum0_v = Q6_Vsf_vadd_VsfVsf(vsqsum0_v, Q6_V_vror_VR(vsqsum0_v, 16));
        vsqsum1_v = Q6_Vsf_vadd_VsfVsf(vsqsum1_v, Q6_V_vror_VR(vsqsum1_v, 16));
        vsqsum0_v = Q6_Vsf_vadd_VsfVsf(vsqsum0_v, Q6_V_vror_VR(vsqsum0_v, 8));
        vsqsum1_v = Q6_Vsf_vadd_VsfVsf(vsqsum1_v, Q6_V_vror_VR(vsqsum1_v, 8));
        vsqsum0_v = Q6_Vsf_vadd_VsfVsf(vsqsum0_v, Q6_V_vror_VR(vsqsum0_v, 4));
        vsqsum1_v = Q6_Vsf_vadd_VsfVsf(vsqsum1_v, Q6_V_vror_VR(vsqsum1_v, 4));

        union { HVX_Vector v; float f[32]; } usqsum0 = { .v = vsqsum0_v };
        union { HVX_Vector v; float f[32]; } usqsum1 = { .v = vsqsum1_v };
        sqsum0 = usqsum0.f[0];
        sqsum1 = usqsum1.f[0];

        if (!rms_norm) {
            HVX_Vector vsum0_v = Q6_Vsf_vadd_VsfVsf(v0_sum0, v0_sum1);
            HVX_Vector vsum1_v = Q6_Vsf_vadd_VsfVsf(v1_sum0, v1_sum1);
            vsum0_v = Q6_Vsf_vadd_VsfVsf(vsum0_v, Q6_V_vror_VR(vsum0_v, 64));
            vsum1_v = Q6_Vsf_vadd_VsfVsf(vsum1_v, Q6_V_vror_VR(vsum1_v, 64));
            vsum0_v = Q6_Vsf_vadd_VsfVsf(vsum0_v, Q6_V_vror_VR(vsum0_v, 32));
            vsum1_v = Q6_Vsf_vadd_VsfVsf(vsum1_v, Q6_V_vror_VR(vsum1_v, 32));
            vsum0_v = Q6_Vsf_vadd_VsfVsf(vsum0_v, Q6_V_vror_VR(vsum0_v, 16));
            vsum1_v = Q6_Vsf_vadd_VsfVsf(vsum1_v, Q6_V_vror_VR(vsum1_v, 16));
            vsum0_v = Q6_Vsf_vadd_VsfVsf(vsum0_v, Q6_V_vror_VR(vsum0_v, 8));
            vsum1_v = Q6_Vsf_vadd_VsfVsf(vsum1_v, Q6_V_vror_VR(vsum1_v, 8));
            vsum0_v = Q6_Vsf_vadd_VsfVsf(vsum0_v, Q6_V_vror_VR(vsum0_v, 4));
            vsum1_v = Q6_Vsf_vadd_VsfVsf(vsum1_v, Q6_V_vror_VR(vsum1_v, 4));

            union { HVX_Vector v; float f[32]; } usum0 = { .v = vsum0_v };
            union { HVX_Vector v; float f[32]; } usum1 = { .v = vsum1_v };
            sum0 = usum0.f[0];
            sum1 = usum1.f[0];
        }
    }

    for (int i = vec_end; i < head_dim; ++i) {
        float val0 = (float)in0_ptr[i];
        float val1 = (float)in1_ptr[i];
        if (!rms_norm) {
            sum0 += val0;
            sum1 += val1;
        }
        sqsum0 += val0 * val0;
        sqsum1 += val1 * val1;
    }

    float mean0 = 0.0f, mean1 = 0.0f;
    if (!rms_norm) {
        mean0 = sum0 / head_dim;
        mean1 = sum1 / head_dim;
    }
    float variance0 = sqsum0 / head_dim;
    float variance1 = sqsum1 / head_dim;
    if (!rms_norm) {
        variance0 -= mean0 * mean0;
        variance1 -= mean1 * mean1;
    }
    float inv_std0 = 1.0f / __builtin_sqrtf(variance0 + epsilon);
    float inv_std1 = 1.0f / __builtin_sqrtf(variance1 + epsilon);

    union { float f; int32_t i; } u_mean0 = { .f = mean0 };
    union { float f; int32_t i; } u_mean1 = { .f = mean1 };
    union { float f; int32_t i; } u_inv_std0 = { .f = inv_std0 };
    union { float f; int32_t i; } u_inv_std1 = { .f = inv_std1 };
    HVX_Vector vmean0 = Q6_V_vsplat_R(u_mean0.i);
    HVX_Vector vmean1 = Q6_V_vsplat_R(u_mean1.i);
    HVX_Vector vinv_std0 = Q6_V_vsplat_R(u_inv_std0.i);
    HVX_Vector vinv_std1 = Q6_V_vsplat_R(u_inv_std1.i);

    vec_in0_ptr = in0_ptr;
    vec_in1_ptr = in1_ptr;
    __fp16* vec_out0_ptr = out0_ptr;
    __fp16* vec_out1_ptr = out1_ptr;
    const float* vec_gamma_ptr = gamma_ptr;
    for (int i = 0; i < fullUnit; ++i, vec_in0_ptr += 64, vec_in1_ptr += 64,
                                      vec_out0_ptr += 64, vec_out1_ptr += 64, vec_gamma_ptr += 64) {
        HVX_Vector s0_hf = *((const HVX_UVector*)vec_in0_ptr);
        HVX_Vector s1_hf = *((const HVX_UVector*)vec_in1_ptr);
        HVX_VectorPair s0_sf = Q6_Wsf_vcvt_Vhf(s0_hf);
        HVX_VectorPair s1_sf = Q6_Wsf_vcvt_Vhf(s1_hf);

        HVX_Vector s00 = Q6_V_lo_W(s0_sf);
        HVX_Vector s01 = Q6_V_hi_W(s0_sf);
        HVX_Vector s10 = Q6_V_lo_W(s1_sf);
        HVX_Vector s11 = Q6_V_hi_W(s1_sf);

        HVX_Vector out00;
        HVX_Vector out01;
        HVX_Vector out10;
        HVX_Vector out11;
        if (rms_norm) {
            out00 = Q6_Vsf_vmpy_VsfVsf(s00, vinv_std0);
            out01 = Q6_Vsf_vmpy_VsfVsf(s01, vinv_std0);
            out10 = Q6_Vsf_vmpy_VsfVsf(s10, vinv_std1);
            out11 = Q6_Vsf_vmpy_VsfVsf(s11, vinv_std1);
        } else {
            out00 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(s00, vmean0), vinv_std0);
            out01 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(s01, vmean0), vinv_std0);
            out10 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(s10, vmean1), vinv_std1);
            out11 = Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vsub_VsfVsf(s11, vmean1), vinv_std1);
        }

        HVX_Vector vg0 = *((const HVX_UVector*)vec_gamma_ptr);
        HVX_Vector vg1 = *((const HVX_UVector*)(vec_gamma_ptr + 32));
        HVX_VectorPair g_deal = Q6_W_vdeal_VVR(vg1, vg0, -4);
        HVX_Vector g0 = Q6_V_lo_W(g_deal);
        HVX_Vector g1 = Q6_V_hi_W(g_deal);

        out00 = Q6_Vsf_vmpy_VsfVsf(out00, g0);
        out01 = Q6_Vsf_vmpy_VsfVsf(out01, g1);
        out10 = Q6_Vsf_vmpy_VsfVsf(out10, g0);
        out11 = Q6_Vsf_vmpy_VsfVsf(out11, g1);

        *((HVX_UVector*)vec_out0_ptr) = Q6_Vhf_vcvt_VsfVsf(out00, out01);
        *((HVX_UVector*)vec_out1_ptr) = Q6_Vhf_vcvt_VsfVsf(out10, out11);
    }

    for (int i = vec_end; i < head_dim; ++i) {
        float norm_val0 = ((float)in0_ptr[i] - mean0) * inv_std0;
        float norm_val1 = ((float)in1_ptr[i] - mean1) * inv_std1;
        float g = gamma_ptr[i];
        out0_ptr[i] = (__fp16)(norm_val0 * g);
        out1_ptr[i] = (__fp16)(norm_val1 * g);
    }
}

static inline void compute_layernorm_head_fp16(__fp16* out_base, const __fp16* in_base, const float* gamma,
                                               int headNumber, int head_dim, float epsilon, int rms_norm) {
    int h = 0;
    for (; h + 2 <= headNumber; h += 2) {
        compute_layernorm_two_heads_fp16(out_base + (size_t)h * head_dim,
                                         out_base + (size_t)(h + 1) * head_dim,
                                         in_base + (size_t)h * head_dim,
                                         in_base + (size_t)(h + 1) * head_dim,
                                         gamma, head_dim, epsilon, rms_norm);
    }
    for (; h < headNumber; ++h) {
        compute_layernorm_one_head_fp16(out_base + (size_t)h * head_dim,
                                        in_base + (size_t)h * head_dim,
                                        gamma, head_dim, epsilon, rms_norm);
    }
}

AEEResult htp_ops_rope_fuse_layernorm(uint8_t* q_out_ptr, uint8_t* q_in_ptr,
                                      uint8_t* k_out_ptr, uint8_t* k_in_ptr,
                                      uint8_t* cos_even_ptr, uint8_t* cos_odd_ptr,
                                      uint8_t* sin_even_ptr, uint8_t* sin_odd_ptr,
                                      uint8_t* q_gamma_ptr,
                                      uint8_t* k_gamma_ptr,
                                      int32_t batch_seq, int32_t num_head, int32_t kv_num_head,
                                      int32_t head_dim, int32_t rope_dim,
                                      float q_epsilon, float k_epsilon,
                                      int32_t q_rms_norm, int32_t k_rms_norm,
                                      int32_t input_c4) {

    if (head_dim % 2 != 0) return AEE_EBADPARM;
    if (head_dim <= 0 || batch_seq <= 0 || num_head <= 0 || kv_num_head <= 0) {
        return AEE_EBADPARM;
    }
    if (q_out_ptr == NULL || q_in_ptr == NULL || k_out_ptr == NULL || k_in_ptr == NULL) {
        return AEE_EBADPARM;
    }
    if (cos_even_ptr == NULL || sin_even_ptr == NULL) {
        return AEE_EBADPARM;
    }

    int half_head_dim = head_dim / 2;
    int rope_half_head_dim = rope_dim / 2;
    if (rope_half_head_dim > half_head_dim) rope_half_head_dim = half_head_dim;

    const __fp16* q_in = (const __fp16*)q_in_ptr;
    __fp16* q_norm_buf = (__fp16*)q_out_ptr;
    const __fp16* k_in = (const __fp16*)k_in_ptr;
    __fp16* k_norm_buf = (__fp16*)k_out_ptr;
    const __fp16* cos_even = (const __fp16*)cos_even_ptr;
    const __fp16* cos_odd = (const __fp16*)cos_odd_ptr;
    const __fp16* sin_even = (const __fp16*)sin_even_ptr;
    const __fp16* sin_odd = (const __fp16*)sin_odd_ptr;
    const float* q_gamma = (const float*)q_gamma_ptr;
    const float* k_gamma = (const float*)k_gamma_ptr;

    uint8_t* vtcm_ptr = (uint8_t*)vtcm_manager_get_vtcm_base();
    if (vtcm_ptr == NULL) {
        return AEE_ENOMEMORY;
    }

    float* q_gamma_vtcm = NULL;
    float* k_gamma_vtcm = NULL;

    if (q_gamma) {
        q_gamma_vtcm = (float*)vtcm_seq_alloc(&vtcm_ptr, head_dim * sizeof(float));
        memcpy(q_gamma_vtcm, q_gamma, head_dim * sizeof(float));
    }

    if (k_gamma) {
        k_gamma_vtcm = (float*)vtcm_seq_alloc(&vtcm_ptr, head_dim * sizeof(float));
        memcpy(k_gamma_vtcm, k_gamma, head_dim * sizeof(float));
    }

    const int n_tasks = htp_ops_rope_pick_batch_task_count(batch_seq);
    const int worker_slots = n_tasks <= 1 ? 1 : (g_max_num_workers > 0 ? (int)g_max_num_workers : n_tasks);
    const size_t worker_vtcm_bytes =
        htp_ops_rope_align_vtcm_bytes((size_t)num_head * head_dim * sizeof(__fp16)) +
        htp_ops_rope_align_vtcm_bytes((size_t)kv_num_head * head_dim * sizeof(__fp16)) +
        htp_ops_rope_align_vtcm_bytes((size_t)2 * head_dim * sizeof(__fp16));
    uint8_t* workspace_base = vtcm_seq_alloc(&vtcm_ptr, (size_t)worker_slots * worker_vtcm_bytes);

    if (n_tasks <= 1) {
        uint8_t* worker_vtcm_ptr = workspace_base;
        __fp16* q_mid = (__fp16*)vtcm_seq_alloc(&worker_vtcm_ptr, (size_t)num_head * head_dim * sizeof(__fp16));
        __fp16* k_mid = (__fp16*)vtcm_seq_alloc(&worker_vtcm_ptr, (size_t)kv_num_head * head_dim * sizeof(__fp16));
        __fp16* trig_mid = (__fp16*)vtcm_seq_alloc(&worker_vtcm_ptr, (size_t)2 * head_dim * sizeof(__fp16));
        for (int i = 0; i < batch_seq; ++i) {
            htp_ops_rope_fuse_layernorm_process_one_seq(i,
                                                        q_in, q_norm_buf,
                                                        k_in, k_norm_buf,
                                                        cos_even, cos_odd,
                                                        sin_even, sin_odd,
                                                        q_gamma_vtcm, k_gamma_vtcm,
                                                        num_head, kv_num_head, head_dim,
                                                        half_head_dim, rope_half_head_dim, rope_dim,
                                                        q_epsilon, k_epsilon,
                                                        q_rms_norm, k_rms_norm,
                                                        q_mid, k_mid, trig_mid,
                                                        input_c4, batch_seq);
        }
    } else {
        HtpOpsRopeFuseLayernormBatchTaskState state = {};
        state.task_id = 0;
        state.n_tasks = n_tasks;
        state.batch_seq = batch_seq;
        state.num_head = num_head;
        state.kv_num_head = kv_num_head;
        state.head_dim = head_dim;
        state.half_head_dim = half_head_dim;
        state.rope_half_head_dim = rope_half_head_dim;
        state.rope_dim = rope_dim;
        state.q_epsilon = q_epsilon;
        state.k_epsilon = k_epsilon;
        state.q_rms_norm = q_rms_norm;
        state.k_rms_norm = k_rms_norm;
        state.worker_vtcm_bytes = worker_vtcm_bytes;
        state.workspace_base = workspace_base;
        state.q_in = q_in;
        state.q_out = q_norm_buf;
        state.k_in = k_in;
        state.k_out = k_norm_buf;
        state.cos_even = cos_even;
        state.cos_odd = cos_odd;
        state.sin_even = sin_even;
        state.sin_odd = sin_odd;
        state.q_gamma = q_gamma_vtcm;
        state.k_gamma = k_gamma_vtcm;
        state.input_c4 = input_c4;

        worker_pool_job_t job;
        job.fptr = htp_ops_rope_fuse_layernorm_batch_worker;
        job.dptr = &state;

        worker_pool_synctoken_init(&(state.sync_ctx), n_tasks);
        for (int i = 0; i < n_tasks; ++i) {
            worker_pool_submit(NULL, job);
        }
        worker_pool_synctoken_wait(&(state.sync_ctx));
    }

    return AEE_SUCCESS;
}

}  // extern "C"

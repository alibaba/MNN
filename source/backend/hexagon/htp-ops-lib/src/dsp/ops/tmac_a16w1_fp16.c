#include <AEEStdErr.h>
#include <stdint.h>
#include <string.h>

#include "dsp/ops.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_utils.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"

static inline float load_pack64_fp16(const __fp16* src, int m, int pos, int ic_index) {
    const int ic_pack_index = ic_index >> 6;
    const int ic_inner = ic_index & 63;
    return (float)src[((size_t)ic_pack_index * m + pos) * 64 + ic_inner];
}

typedef struct {
    __fp16* dst_h;
    const __fp16* src_h;
    const uint8_t* weight;
    const float* scale;
    const __fp16* bias_h;
    const float* input_sums;
    int m;
    int pos;
    int ic;
    int oc;
    int scale_block_num;
    int scale_asymmetric;
    int relu;
    int relu6;
    int block_size;
    int weight_block_bytes;
    worker_synctoken_t sync_ctx;
} TmacA16W1TaskState;

typedef struct {
    TmacA16W1TaskState* state;
    int oc_start;
    int oc_end;
} TmacA16W1Task;

typedef struct {
    __fp16* dst_h;
    const __fp16* src_h;
    const uint8_t* weight;
    const float* scale;
    const __fp16* bias_h;
    const float* input_sums;
    int m;
    int pos;
    int ic;
    int oc;
    int scale_block_num;
    int scale_asymmetric;
    int relu;
    int relu6;
    int block_size;
    int weight_block_bytes;
    const __fp16* table_lut_h;
    worker_synctoken_t sync_ctx;
} TmacA16W1HvxState;

typedef struct {
    TmacA16W1HvxState* state;
    int oc_pack_start;
    int oc_pack_end;
} TmacA16W1HvxTask;

typedef struct {
    __fp16* table_lut_h;
    const __fp16* src_h;
    int m;
    int pos;
    int block_size;
    int weight_block_bytes;
    int scale_block_num;
    int total_weight_bytes;
    worker_synctoken_t sync_ctx;
} TmacBuildTableState;

typedef struct {
    TmacBuildTableState* state;
    int start;
    int end;
    float* input_sums;
} TmacBuildTableTask;

static void compute_tmac_oc_range(TmacA16W1TaskState* state, int oc_start, int oc_end) {
    const int oc_pack = 64;
    const float relu6_value = 6.0f;
    for (int oc_index = oc_start; oc_index < oc_end; ++oc_index) {
        const int oc_pack_index = oc_index / oc_pack;
        const int oc_inner = oc_index - oc_pack_index * oc_pack;
        float acc = 0.0f;
        if (oc_index < state->oc && state->bias_h != 0) {
            acc = (float)state->bias_h[oc_pack_index * oc_pack + oc_inner];
        }
        if (oc_index < state->oc) {
            const uint8_t* weight_oc = state->weight + (size_t)oc_index * state->scale_block_num * state->weight_block_bytes;
            const float* scale_oc = state->scale + (size_t)oc_index * state->scale_block_num * (state->scale_asymmetric ? 2 : 1);
            for (int block = 0; block < state->scale_block_num; ++block) {
                float selected_sum = 0.0f;
                const int ic_base = block * state->block_size;
                const uint8_t* weight_block = weight_oc + (size_t)block * state->weight_block_bytes;
                for (int byte = 0; byte < state->weight_block_bytes; ++byte) {
                    const uint8_t bits = weight_block[byte];
                    const int ic_byte_base = ic_base + byte * 8;
                    if (bits == 0) {
                        continue;
                    }
                    if (bits == 0xff) {
                        for (int bit = 0; bit < 8; ++bit) {
                            selected_sum += load_pack64_fp16(state->src_h, state->m, state->pos, ic_byte_base + bit);
                        }
                        continue;
                    }
                    for (int bit = 0; bit < 8; ++bit) {
                        if (bits & (0x80 >> bit)) {
                            selected_sum += load_pack64_fp16(state->src_h, state->m, state->pos, ic_byte_base + bit);
                        }
                    }
                }
                const float input_sum = state->input_sums[block];
                if (state->scale_asymmetric) {
                    const float min_value = scale_oc[2 * block + 0];
                    const float alpha = scale_oc[2 * block + 1];
                    acc += selected_sum * alpha + input_sum * (min_value - alpha);
                } else {
                    acc += (selected_sum - input_sum) * scale_oc[block];
                }
            }
        }
        if (state->relu || state->relu6) {
            if (acc < 0.0f) {
                acc = 0.0f;
            }
            if (state->relu6 && acc > relu6_value) {
                acc = relu6_value;
            }
        }
        state->dst_h[((size_t)oc_pack_index * state->m + state->pos) * oc_pack + oc_inner] = (__fp16)acc;
    }
}

static void compute_tmac_worker(void* data, int worker_index) {
    (void)worker_index;
    TmacA16W1Task* task = (TmacA16W1Task*)data;
    compute_tmac_oc_range(task->state, task->oc_start, task->oc_end);
    worker_pool_synctoken_jobdone(&task->state->sync_ctx);
}

static inline void tmac_make_bias_vecs(float* bias_even_f, float* bias_odd_f, const __fp16* bias_h,
                                       int oc_base, int oc) {
    for (int lane = 0; lane < 32; ++lane) {
        const int even_oc_index = oc_base + lane * 2;
        const int odd_oc_index = even_oc_index + 1;
        bias_even_f[lane] = (even_oc_index < oc && bias_h != 0) ? (float)bias_h[even_oc_index] : 0.0f;
        bias_odd_f[lane] = (odd_oc_index < oc && bias_h != 0) ? (float)bias_h[odd_oc_index] : 0.0f;
    }
}

static inline void tmac_store_subset_lut_h(__fp16* lut, const __fp16* src) {
    uint32_t* lut32 = (uint32_t*)lut;
    const __fp16 s3 = src[0];
    const __fp16 s2 = src[1];
    const __fp16 s1 = src[2];
    const __fp16 s0 = src[3];
    const __fp16 s01 = s0 + s1;
    const __fp16 s02 = s0 + s2;
    const __fp16 s12 = s1 + s2;
    const __fp16 s012 = s01 + s2;
    const __fp16 s03 = s0 + s3;
    const __fp16 s13 = s1 + s3;
    const __fp16 s013 = s01 + s3;
    const __fp16 s23 = s2 + s3;
    const __fp16 s023 = s02 + s3;
    const __fp16 s123 = s12 + s3;
    const __fp16 s0123 = s012 + s3;
    lut32[0] = 0;
    lut32[1] = (uint32_t)(*(const uint16_t*)&s0);
    lut32[2] = (uint32_t)(*(const uint16_t*)&s1);
    lut32[3] = (uint32_t)(*(const uint16_t*)&s01);
    lut32[4] = (uint32_t)(*(const uint16_t*)&s2);
    lut32[5] = (uint32_t)(*(const uint16_t*)&s02);
    lut32[6] = (uint32_t)(*(const uint16_t*)&s12);
    lut32[7] = (uint32_t)(*(const uint16_t*)&s012);
    lut32[8] = (uint32_t)(*(const uint16_t*)&s3);
    lut32[9] = (uint32_t)(*(const uint16_t*)&s03);
    lut32[10] = (uint32_t)(*(const uint16_t*)&s13);
    lut32[11] = (uint32_t)(*(const uint16_t*)&s013);
    lut32[12] = (uint32_t)(*(const uint16_t*)&s23);
    lut32[13] = (uint32_t)(*(const uint16_t*)&s023);
    lut32[14] = (uint32_t)(*(const uint16_t*)&s123);
    lut32[15] = (uint32_t)(*(const uint16_t*)&s0123);
}

static inline void tmac_build_table_lut_h_range(TmacBuildTableState* state, int start, int end,
                                                float* input_sums) {
    for (int k = start; k < end; ++k) {
        const int block = k / state->weight_block_bytes;
        const int byte = k - block * state->weight_block_bytes;
        const int ic_byte_base = block * state->block_size + byte * 8;
        const __fp16* src = state->src_h + ((size_t)(ic_byte_base >> 6) * state->m + state->pos) * 64 +
                            (ic_byte_base & 63);
        __fp16* lut = state->table_lut_h + (size_t)k * 2 * 64;
        tmac_store_subset_lut_h(lut, src);
        tmac_store_subset_lut_h(lut + 64, src + 4);
        input_sums[block] += (float)src[0] + (float)src[1] + (float)src[2] + (float)src[3] +
                             (float)src[4] + (float)src[5] + (float)src[6] + (float)src[7];
    }
}

static inline void tmac_build_table_lut_h_range_oneblock(TmacBuildTableState* state, int start, int end,
                                                         float* input_sums) {
    float input_sum = 0.0f;
    const __fp16* src = state->src_h + (size_t)start * 8;
    __fp16* lut = state->table_lut_h + (size_t)start * 2 * 64;
    for (int k = start; k < end; ++k) {
        tmac_store_subset_lut_h(lut, src);
        tmac_store_subset_lut_h(lut + 64, src + 4);
        input_sum += (float)src[0] + (float)src[1] + (float)src[2] + (float)src[3] +
                     (float)src[4] + (float)src[5] + (float)src[6] + (float)src[7];
        src += 8;
        lut += 128;
    }
    input_sums[0] += input_sum;
}

static void tmac_build_table_worker(void* data, int worker_index) {
    (void)worker_index;
    TmacBuildTableTask* task = (TmacBuildTableTask*)data;
    if (task->state->scale_block_num == 1 && task->state->m == 1) {
        tmac_build_table_lut_h_range_oneblock(task->state, task->start, task->end, task->input_sums);
    } else {
        tmac_build_table_lut_h_range(task->state, task->start, task->end, task->input_sums);
    }
    worker_pool_synctoken_jobdone(&task->state->sync_ctx);
}

static inline void tmac_build_table_lut_h(__fp16* table_lut_h, const __fp16* src_h,
                                          int m, int pos, int block_size,
                                          int weight_block_bytes, int scale_block_num,
                                          float* input_sums) {
    const int total_weight_bytes = scale_block_num * weight_block_bytes;
    TmacBuildTableState state = {table_lut_h, src_h, m, pos, block_size, weight_block_bytes,
                                 scale_block_num, total_weight_bytes};
    for (int block = 0; block < scale_block_num; ++block) {
        input_sums[block] = 0.0f;
    }
    int task_count = 1;
    if (total_weight_bytes >= 96 && g_max_num_workers > 1) {
        task_count = (int)g_max_num_workers;
        const int table_chunks = (total_weight_bytes + 47) / 48;
        if (task_count > table_chunks) {
            task_count = table_chunks;
        }
    }
    if (task_count <= 1) {
        if (scale_block_num == 1 && m == 1) {
            tmac_build_table_lut_h_range_oneblock(&state, 0, total_weight_bytes, input_sums);
        } else {
            tmac_build_table_lut_h_range(&state, 0, total_weight_bytes, input_sums);
        }
    } else {
        TmacBuildTableTask* tasks = WORKER_POOL_STACK_ALLOC(TmacBuildTableTask, task_count);
        float* partial_sums = WORKER_POOL_STACK_ALLOC(float, task_count * scale_block_num);
        memset(partial_sums, 0, (size_t)task_count * scale_block_num * sizeof(float));
        worker_pool_job_t job;
        job.fptr = tmac_build_table_worker;
        worker_pool_synctoken_init(&state.sync_ctx, task_count);
        for (int t = 0; t < task_count; ++t) {
            tasks[t].state = &state;
            tasks[t].start = total_weight_bytes * t / task_count;
            tasks[t].end = total_weight_bytes * (t + 1) / task_count;
            tasks[t].input_sums = partial_sums + (size_t)t * scale_block_num;
            job.dptr = tasks + t;
            worker_pool_submit(NULL, job);
        }
        worker_pool_synctoken_wait(&state.sync_ctx);
        for (int t = 0; t < task_count; ++t) {
            const float* partial = partial_sums + (size_t)t * scale_block_num;
            for (int block = 0; block < scale_block_num; ++block) {
                input_sums[block] += partial[block];
            }
        }
    }
}

static inline void tmac_accumulate_lookup_h_to_f32(HVX_Vector* selected00, HVX_Vector* selected01,
                                                   HVX_Vector* selected10, HVX_Vector* selected11,
                                                   HVX_Vector selected0_h, HVX_Vector selected1_h) {
    HVX_VectorPair selected0_f = Q6_Wsf_vcvt_Vhf(selected0_h);
    *selected00 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(*selected00, Q6_V_lo_W(selected0_f)));
    *selected01 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(*selected01, Q6_V_hi_W(selected0_f)));
    HVX_VectorPair selected1_f = Q6_Wsf_vcvt_Vhf(selected1_h);
    *selected10 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(*selected10, Q6_V_lo_W(selected1_f)));
    *selected11 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(*selected11, Q6_V_hi_W(selected1_f)));
}

static inline HVX_Vector tmac_splat_f32(float value) {
    union {
        float f;
        int i;
    } u;
    u.f = value;
    return Q6_V_vsplat_R(u.i);
}

static inline void tmac_vlut_accumulate_byte(HVX_Vector* chunk0_h, HVX_Vector* chunk1_h,
                                             const uint8_t* weight_byte, const __fp16* table_high_h) {
    HVX_Vector vWeightByte = vmemu(weight_byte);
    HVX_Vector vTableHigh = *((const HVX_Vector*)table_high_h);
    HVX_Vector vTableLow = *((const HVX_Vector*)(table_high_h + 64));
    HVX_VectorPair high_pair = Q6_Wh_vlut16_VbVhR_nomatch(Q6_Vub_vlsr_VubR(vWeightByte, 4),
                                                          vTableHigh, 0);
    HVX_VectorPair low_pair = Q6_Wh_vlut16_VbVhR_nomatch(vWeightByte, vTableLow, 0);
    HVX_Vector selected0_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_V_lo_W(high_pair),
                                                                      Q6_V_lo_W(low_pair)));
    HVX_Vector selected1_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_V_hi_W(high_pair),
                                                                      Q6_V_hi_W(low_pair)));
    *chunk0_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(*chunk0_h, selected0_h));
    *chunk1_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(*chunk1_h, selected1_h));
}

static inline void tmac_vlut_accumulate_4bytes(HVX_Vector* chunk0_h, HVX_Vector* chunk1_h,
                                               const uint8_t* weight_byte, const __fp16* table_high_h) {
    HVX_Vector vWeight0 = vmemu(weight_byte);
    HVX_Vector vWeight1 = vmemu(weight_byte + 128);
    HVX_Vector vHigh0 = Q6_Vub_vlsr_VubR(vWeight0, 4);
    HVX_Vector vHigh1 = Q6_Vub_vlsr_VubR(vWeight1, 4);

    HVX_Vector vTableHigh0 = *((const HVX_Vector*)table_high_h);
    HVX_Vector vTableLow0 = *((const HVX_Vector*)(table_high_h + 64));
    HVX_Vector vTableHigh1 = *((const HVX_Vector*)(table_high_h + 128));
    HVX_Vector vTableLow1 = *((const HVX_Vector*)(table_high_h + 192));
    HVX_VectorPair high_pair0 = Q6_Wh_vlut16_VbVhR_nomatch(vHigh0, vTableHigh0, 0);
    HVX_VectorPair low_pair0 = Q6_Wh_vlut16_VbVhR_nomatch(vWeight0, vTableLow0, 0);
    HVX_VectorPair high_pair1 = Q6_Wh_vlut16_VbVhR_nomatch(vHigh1, vTableHigh1, 0);
    HVX_VectorPair low_pair1 = Q6_Wh_vlut16_VbVhR_nomatch(vWeight1, vTableLow1, 0);

    HVX_Vector selected0_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_V_lo_W(high_pair0),
                                                                      Q6_V_lo_W(low_pair0)));
    HVX_Vector selected1_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_V_hi_W(high_pair0),
                                                                      Q6_V_hi_W(low_pair0)));
    HVX_Vector next0_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_V_lo_W(high_pair1),
                                                                  Q6_V_lo_W(low_pair1)));
    HVX_Vector next1_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_V_hi_W(high_pair1),
                                                                  Q6_V_hi_W(low_pair1)));
    selected0_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(selected0_h, next0_h));
    selected1_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(selected1_h, next1_h));

    HVX_Vector vWeight2 = vmemu(weight_byte + 256);
    HVX_Vector vWeight3 = vmemu(weight_byte + 384);
    HVX_Vector vHigh2 = Q6_Vub_vlsr_VubR(vWeight2, 4);
    HVX_Vector vHigh3 = Q6_Vub_vlsr_VubR(vWeight3, 4);
    HVX_Vector vTableHigh2 = *((const HVX_Vector*)(table_high_h + 256));
    HVX_Vector vTableLow2 = *((const HVX_Vector*)(table_high_h + 320));
    HVX_Vector vTableHigh3 = *((const HVX_Vector*)(table_high_h + 384));
    HVX_Vector vTableLow3 = *((const HVX_Vector*)(table_high_h + 448));
    HVX_VectorPair high_pair2 = Q6_Wh_vlut16_VbVhR_nomatch(vHigh2, vTableHigh2, 0);
    HVX_VectorPair low_pair2 = Q6_Wh_vlut16_VbVhR_nomatch(vWeight2, vTableLow2, 0);
    HVX_VectorPair high_pair3 = Q6_Wh_vlut16_VbVhR_nomatch(vHigh3, vTableHigh3, 0);
    HVX_VectorPair low_pair3 = Q6_Wh_vlut16_VbVhR_nomatch(vWeight3, vTableLow3, 0);

    next0_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_V_lo_W(high_pair2),
                                                       Q6_V_lo_W(low_pair2)));
    next1_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_V_hi_W(high_pair2),
                                                       Q6_V_hi_W(low_pair2)));
    selected0_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(selected0_h, next0_h));
    selected1_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(selected1_h, next1_h));
    next0_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_V_lo_W(high_pair3),
                                                       Q6_V_lo_W(low_pair3)));
    next1_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_V_hi_W(high_pair3),
                                                       Q6_V_hi_W(low_pair3)));
    *chunk0_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(*chunk0_h,
                                                        Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(selected0_h,
                                                                                                  next0_h))));
    *chunk1_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(*chunk1_h,
                                                        Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(selected1_h,
                                                                                                  next1_h))));
}

static void compute_tmac_hvx_pack_range(TmacA16W1HvxState* state, int oc_pack_start, int oc_pack_end) {
    const int output_oc_pack = 64;
    const int weight_oc_pack = 128;
    const HVX_Vector vZero = Q6_V_vzero();
    const HVX_Vector vRelu6 = Q6_V_vsplat_R(0x40c00000);

    _Alignas(128) float bias_even_f[32];
    _Alignas(128) float bias_odd_f[32];
    const int raw_scale_count = state->oc * state->scale_block_num * (state->scale_asymmetric ? 2 : 1);
    const float* packed_scale = state->scale + raw_scale_count;

    for (int oc_pack_index = oc_pack_start; oc_pack_index < oc_pack_end; ++oc_pack_index) {
        const int oc_base = oc_pack_index * weight_oc_pack;
        const uint8_t* weight_pack = state->weight + (size_t)oc_pack_index *
                                     state->scale_block_num * state->weight_block_bytes * 128;
        l2fetch(weight_pack, 128, 128, state->weight_block_bytes, 0);
        tmac_make_bias_vecs(bias_even_f, bias_odd_f, state->bias_h, oc_base, state->oc);
        HVX_Vector acc00 = vmemu(bias_even_f);
        HVX_Vector acc01 = vmemu(bias_odd_f);
        tmac_make_bias_vecs(bias_even_f, bias_odd_f, state->bias_h, oc_base + output_oc_pack, state->oc);
        HVX_Vector acc10 = vmemu(bias_even_f);
        HVX_Vector acc11 = vmemu(bias_odd_f);

        for (int block = 0; block < state->scale_block_num; ++block) {
            HVX_Vector selected00 = Q6_V_vzero();
            HVX_Vector selected01 = Q6_V_vzero();
            HVX_Vector selected10 = Q6_V_vzero();
            HVX_Vector selected11 = Q6_V_vzero();
            const uint8_t* weight_byte = weight_pack + (size_t)block * state->weight_block_bytes * 128;
            const __fp16* table_high_h = state->table_lut_h +
                (size_t)block * state->weight_block_bytes * 2 * 64;
            int byte = 0;
            for (; byte + 7 < state->weight_block_bytes; byte += 8) {
                HVX_Vector chunk0_a = Q6_V_vzero();
                HVX_Vector chunk1_a = Q6_V_vzero();
                HVX_Vector chunk0_b = Q6_V_vzero();
                HVX_Vector chunk1_b = Q6_V_vzero();
                tmac_vlut_accumulate_4bytes(&chunk0_a, &chunk1_a, weight_byte, table_high_h);
                tmac_vlut_accumulate_4bytes(&chunk0_b, &chunk1_b, weight_byte + 512, table_high_h + 512);
                tmac_accumulate_lookup_h_to_f32(&selected00, &selected01, &selected10, &selected11,
                                                chunk0_a, chunk1_a);
                tmac_accumulate_lookup_h_to_f32(&selected00, &selected01, &selected10, &selected11,
                                                chunk0_b, chunk1_b);
                weight_byte += 1024;
                table_high_h += 1024;
            }
            if (byte < state->weight_block_bytes) {
                HVX_Vector chunk0_h = Q6_V_vzero();
                HVX_Vector chunk1_h = Q6_V_vzero();
                for (; byte < state->weight_block_bytes; ++byte) {
                    tmac_vlut_accumulate_byte(&chunk0_h, &chunk1_h, weight_byte, table_high_h);
                    weight_byte += 128;
                    table_high_h += 128;
                }
                tmac_accumulate_lookup_h_to_f32(&selected00, &selected01, &selected10, &selected11,
                                                chunk0_h, chunk1_h);
            }
            if (block + 1 < state->scale_block_num) {
                const uint8_t* next_weight_block = weight_pack + (size_t)(block + 1) *
                                                   state->weight_block_bytes * 128;
                l2fetch(next_weight_block, 128, 128, state->weight_block_bytes, 0);
            }

            const float* scale_pack = packed_scale + ((size_t)oc_pack_index * state->scale_block_num + block) * 8 * 32;
            HVX_Vector input_sum_vec = tmac_splat_f32(state->input_sums[block]);
            HVX_Vector scaled00 = Q6_Vsf_vmpy_VsfVsf(selected00, *((const HVX_Vector*)(scale_pack + 0 * 32)));
            HVX_Vector scaled01 = Q6_Vsf_vmpy_VsfVsf(selected01, *((const HVX_Vector*)(scale_pack + 2 * 32)));
            HVX_Vector offset00 = Q6_Vsf_vmpy_VsfVsf(input_sum_vec, *((const HVX_Vector*)(scale_pack + 1 * 32)));
            HVX_Vector offset01 = Q6_Vsf_vmpy_VsfVsf(input_sum_vec, *((const HVX_Vector*)(scale_pack + 3 * 32)));
            scaled00 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(scaled00, offset00));
            scaled01 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(scaled01, offset01));
            acc00 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(acc00, scaled00));
            acc01 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(acc01, scaled01));
            HVX_Vector scaled10 = Q6_Vsf_vmpy_VsfVsf(selected10, *((const HVX_Vector*)(scale_pack + 4 * 32)));
            HVX_Vector scaled11 = Q6_Vsf_vmpy_VsfVsf(selected11, *((const HVX_Vector*)(scale_pack + 6 * 32)));
            HVX_Vector offset10 = Q6_Vsf_vmpy_VsfVsf(input_sum_vec, *((const HVX_Vector*)(scale_pack + 5 * 32)));
            HVX_Vector offset11 = Q6_Vsf_vmpy_VsfVsf(input_sum_vec, *((const HVX_Vector*)(scale_pack + 7 * 32)));
            scaled10 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(scaled10, offset10));
            scaled11 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(scaled11, offset11));
            acc10 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(acc10, scaled10));
            acc11 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(acc11, scaled11));
        }

        if (state->relu || state->relu6) {
            acc00 = Q6_Vsf_vmax_VsfVsf(acc00, vZero);
            acc01 = Q6_Vsf_vmax_VsfVsf(acc01, vZero);
            acc10 = Q6_Vsf_vmax_VsfVsf(acc10, vZero);
            acc11 = Q6_Vsf_vmax_VsfVsf(acc11, vZero);
            if (state->relu6) {
                HVX_VectorPred qGtRelu6_00 = Q6_Q_vcmp_gt_VsfVsf(acc00, vRelu6);
                HVX_VectorPred qGtRelu6_01 = Q6_Q_vcmp_gt_VsfVsf(acc01, vRelu6);
                HVX_VectorPred qGtRelu6_10 = Q6_Q_vcmp_gt_VsfVsf(acc10, vRelu6);
                HVX_VectorPred qGtRelu6_11 = Q6_Q_vcmp_gt_VsfVsf(acc11, vRelu6);
                acc00 = Q6_V_vmux_QVV(qGtRelu6_00, vRelu6, acc00);
                acc01 = Q6_V_vmux_QVV(qGtRelu6_01, vRelu6, acc01);
                acc10 = Q6_V_vmux_QVV(qGtRelu6_10, vRelu6, acc10);
                acc11 = Q6_V_vmux_QVV(qGtRelu6_11, vRelu6, acc11);
            }
        }
        const int output_pack_index = oc_pack_index * 2;
        HVX_Vector acc0 = Q6_Vhf_vcvt_VsfVsf(acc00, acc01);
        vmemu(state->dst_h + ((size_t)output_pack_index * state->m + state->pos) * output_oc_pack) = acc0;
        if (oc_base + output_oc_pack < ((state->oc + output_oc_pack - 1) / output_oc_pack) * output_oc_pack) {
            HVX_Vector acc1 = Q6_Vhf_vcvt_VsfVsf(acc10, acc11);
            vmemu(state->dst_h + ((size_t)(output_pack_index + 1) * state->m + state->pos) * output_oc_pack) = acc1;
        }
    }
}

static void compute_tmac_hvx_worker(void* data, int worker_index) {
    (void)worker_index;
    TmacA16W1HvxTask* task = (TmacA16W1HvxTask*)data;
    compute_tmac_hvx_pack_range(task->state, task->oc_pack_start, task->oc_pack_end);
    worker_pool_synctoken_jobdone(&task->state->sync_ctx);
}

static int compute_tmac_hvx(uint8_t *dst, const uint8_t *src, const uint8_t *weight, const float *scale,
                            const uint8_t *bias, int m, int ic, int oc, int scale_block_num,
                            int scale_asymmetric, int relu, int relu6, int output_bytes) {
    if ((ic & 63) != 0) {
        return AEE_EUNSUPPORTED;
    }
    const int output_oc_pack = 64;
    const int weight_oc_pack = 128;
    const int oc_round = ((oc + output_oc_pack - 1) / output_oc_pack) * output_oc_pack;
    const int output_oc_pack_count = oc_round / output_oc_pack;
    const int weight_oc_pack_count = (oc + weight_oc_pack - 1) / weight_oc_pack;
    const int block_size = ic / scale_block_num;
    const int weight_block_bytes = block_size >> 3;
    const int needed_output_bytes = output_oc_pack_count * m * output_oc_pack * (int)sizeof(__fp16);
    if (output_bytes > 0 && needed_output_bytes > output_bytes) {
        return AEE_EBADPARM;
    }

    const __fp16* src_h = (const __fp16*)src;
    __fp16* dst_h = (__fp16*)dst;
    const __fp16* bias_h = (const __fp16*)bias;
    float input_sums[scale_block_num];
    const size_t table_lut_count = (size_t)scale_block_num * weight_block_bytes * 2 * 64;
    uint8_t* vtcm_ptr = (uint8_t*)vtcm_manager_get_vtcm_base();
    if (vtcm_ptr == 0) {
        return AEE_EUNSUPPORTED;
    }
    __fp16* table_lut_h = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, table_lut_count * sizeof(__fp16));

    for (int pos = 0; pos < m; ++pos) {
        tmac_build_table_lut_h(table_lut_h, src_h, m, pos, block_size, weight_block_bytes, scale_block_num,
                               input_sums);
        int task_count = 1;
        if (weight_oc_pack_count >= 2 && g_max_num_workers > 1) {
            task_count = (int)g_max_num_workers;
            if (task_count > weight_oc_pack_count) {
                task_count = weight_oc_pack_count;
            }
        }
        TmacA16W1HvxState state = {dst_h, src_h, weight, scale, bias_h, input_sums, m, pos, ic, oc,
                                   scale_block_num, scale_asymmetric, relu, relu6, block_size,
                                   weight_block_bytes, table_lut_h};
        if (task_count <= 1) {
            compute_tmac_hvx_pack_range(&state, 0, weight_oc_pack_count);
        } else {
            TmacA16W1HvxTask* tasks = WORKER_POOL_STACK_ALLOC(TmacA16W1HvxTask, task_count);
            worker_pool_job_t job;
            job.fptr = compute_tmac_hvx_worker;
            worker_pool_synctoken_init(&state.sync_ctx, task_count);
            for (int t = 0; t < task_count; ++t) {
                tasks[t].state = &state;
                tasks[t].oc_pack_start = weight_oc_pack_count * t / task_count;
                tasks[t].oc_pack_end = weight_oc_pack_count * (t + 1) / task_count;
                job.dptr = tasks + t;
                worker_pool_submit(NULL, job);
            }
            worker_pool_synctoken_wait(&state.sync_ctx);
        }
    }
    return AEE_SUCCESS;
}

int scalar_tmac_a16w1_fp16(uint8_t *dst, const uint8_t *src, const uint8_t *weight, const float *scale,
                           const uint8_t *bias, int m, int ic, int oc, int scale_block_num,
                           int scale_asymmetric, int relu, int relu6, int output_bytes) {
    if (dst == 0 || src == 0 || weight == 0 || scale == 0 || m <= 0 || ic <= 0 || oc <= 0 ||
        scale_block_num <= 0 || (ic % scale_block_num) != 0) {
        return AEE_EBADPARM;
    }
    const int block_size = ic / scale_block_num;
    if ((block_size & 7) != 0) {
        return AEE_EBADPARM;
    }

    const __fp16* src_h = (const __fp16*)src;
    __fp16* dst_h = (__fp16*)dst;
    const __fp16* bias_h = (const __fp16*)bias;
    const int oc_pack = 64;
    const int oc_round = ((oc + oc_pack - 1) / oc_pack) * oc_pack;
    const int weight_block_bytes = block_size >> 3;
    float input_sums[scale_block_num];

    const int needed_output_bytes = ((oc_round / oc_pack) * m * oc_pack) * (int)sizeof(__fp16);
    if (output_bytes > 0 && needed_output_bytes > output_bytes) {
        return AEE_EBADPARM;
    }

    int hvx_ret = compute_tmac_hvx(dst, src, weight, scale, bias, m, ic, oc, scale_block_num,
                                   scale_asymmetric, relu, relu6, output_bytes);
    if (hvx_ret == AEE_SUCCESS) {
        return AEE_SUCCESS;
    }

    for (int pos = 0; pos < m; ++pos) {
        for (int block = 0; block < scale_block_num; ++block) {
            float input_sum = 0.0f;
            const int ic_base = block * block_size;
            for (int i = 0; i < block_size; ++i) {
                input_sum += load_pack64_fp16(src_h, m, pos, ic_base + i);
            }
            input_sums[block] = input_sum;
        }

        int task_count = 1;
        if (oc_round >= 128 && g_max_num_workers > 1) {
            task_count = (int)g_max_num_workers;
            if (task_count > oc_round / 64) {
                task_count = oc_round / 64;
            }
        }
        if (task_count <= 1) {
            TmacA16W1TaskState state = {dst_h, src_h, weight, scale, bias_h, input_sums, m, pos, ic, oc,
                                        scale_block_num, scale_asymmetric, relu, relu6, block_size,
                                        weight_block_bytes};
            compute_tmac_oc_range(&state, 0, oc_round);
        } else {
            TmacA16W1TaskState state = {dst_h, src_h, weight, scale, bias_h, input_sums, m, pos, ic, oc,
                                        scale_block_num, scale_asymmetric, relu, relu6, block_size,
                                        weight_block_bytes};
            TmacA16W1Task* tasks = WORKER_POOL_STACK_ALLOC(TmacA16W1Task, task_count);
            worker_pool_job_t job;
            job.fptr = compute_tmac_worker;
            worker_pool_synctoken_init(&state.sync_ctx, task_count);
            for (int t = 0; t < task_count; ++t) {
                int start_pack = (oc_round / 64) * t / task_count;
                int end_pack = (oc_round / 64) * (t + 1) / task_count;
                tasks[t].state = &state;
                tasks[t].oc_start = start_pack * 64;
                tasks[t].oc_end = end_pack * 64;
                job.dptr = tasks + t;
                worker_pool_submit(NULL, job);
            }
            worker_pool_synctoken_wait(&state.sync_ctx);
        }
    }
    return AEE_SUCCESS;
}

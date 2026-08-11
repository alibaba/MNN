#include <AEEStdErr.h>
#include <stdint.h>
#include <string.h>

#include "dsp/ops.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_utils.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"

typedef struct {
    __fp16* dst_h;
    const __fp16* src_h;
    const uint8_t* weight;
    const float* scale;
    const float* bias;
    int m;
    int pos;
    int ic;
    int oc;
    int scale_block_num;
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
} TmacBuildTableTask;

typedef struct {
    __fp16* table_lut_h0;
    __fp16* table_lut_h1;
    const __fp16* src_h;
    int block_size;
    int weight_block_bytes;
    int total_weight_bytes;
    worker_synctoken_t sync_ctx;
} TmacBuildTableM2State;

typedef struct {
    TmacBuildTableM2State* state;
    int start;
    int end;
} TmacBuildTableM2Task;

static const uint32_t tmac_lut_word_mask[16][VLEN_WORD] __attribute__((aligned(VLEN))) = {
    [1] = {[1] = 0xffffffffu},
    [2] = {[2] = 0xffffffffu},
    [3] = {[3] = 0xffffffffu},
    [4] = {[4] = 0xffffffffu},
    [5] = {[5] = 0xffffffffu},
    [6] = {[6] = 0xffffffffu},
    [7] = {[7] = 0xffffffffu},
    [8] = {[8] = 0xffffffffu},
    [9] = {[9] = 0xffffffffu},
    [10] = {[10] = 0xffffffffu},
    [11] = {[11] = 0xffffffffu},
    [12] = {[12] = 0xffffffffu},
    [13] = {[13] = 0xffffffffu},
    [14] = {[14] = 0xffffffffu},
    [15] = {[15] = 0xffffffffu}
};

static inline HVX_Vector tmac_hadd_h(HVX_Vector a, HVX_Vector b) {
    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(a, b));
}

static inline HVX_Vector tmac_lut_set_word_h(HVX_Vector lut, HVX_Vector value_h,
                                             int lane) {
    HVX_Vector value_low_h = Q6_V_vand_VV(value_h, Q6_V_vsplat_R(0x0000ffff));
    HVX_Vector mask = *((const HVX_Vector*)tmac_lut_word_mask[lane]);
    return Q6_V_vor_VV(lut, Q6_V_vand_VV(value_low_h, mask));
}

static inline void tmac_store_subset_lut_h(__fp16* lut, const __fp16* src) {
    HVX_Vector s3 = Q6_Vh_vsplat_R(fp16_to_bits(src + 0));
    HVX_Vector s2 = Q6_Vh_vsplat_R(fp16_to_bits(src + 1));
    HVX_Vector s1 = Q6_Vh_vsplat_R(fp16_to_bits(src + 2));
    HVX_Vector s0 = Q6_Vh_vsplat_R(fp16_to_bits(src + 3));
    HVX_Vector s01 = tmac_hadd_h(s0, s1);
    HVX_Vector s02 = tmac_hadd_h(s0, s2);
    HVX_Vector s12 = tmac_hadd_h(s1, s2);
    HVX_Vector s012 = tmac_hadd_h(s01, s2);
    HVX_Vector s03 = tmac_hadd_h(s0, s3);
    HVX_Vector s13 = tmac_hadd_h(s1, s3);
    HVX_Vector s013 = tmac_hadd_h(s01, s3);
    HVX_Vector s23 = tmac_hadd_h(s2, s3);
    HVX_Vector s023 = tmac_hadd_h(s02, s3);
    HVX_Vector s123 = tmac_hadd_h(s12, s3);
    HVX_Vector s0123 = tmac_hadd_h(s012, s3);

    HVX_Vector v = Q6_V_vzero();
    v = tmac_lut_set_word_h(v, s0, 1);
    v = tmac_lut_set_word_h(v, s1, 2);
    v = tmac_lut_set_word_h(v, s01, 3);
    v = tmac_lut_set_word_h(v, s2, 4);
    v = tmac_lut_set_word_h(v, s02, 5);
    v = tmac_lut_set_word_h(v, s12, 6);
    v = tmac_lut_set_word_h(v, s012, 7);
    v = tmac_lut_set_word_h(v, s3, 8);
    v = tmac_lut_set_word_h(v, s03, 9);
    v = tmac_lut_set_word_h(v, s13, 10);
    v = tmac_lut_set_word_h(v, s013, 11);
    v = tmac_lut_set_word_h(v, s23, 12);
    v = tmac_lut_set_word_h(v, s023, 13);
    v = tmac_lut_set_word_h(v, s123, 14);
    v = tmac_lut_set_word_h(v, s0123, 15);
    vmem(lut) = v;
}

static inline void tmac_build_table_lut_h_range(TmacBuildTableState* state, int start, int end) {
    for (int k = start; k < end; ++k) {
        const int block = k / state->weight_block_bytes;
        const int byte = k - block * state->weight_block_bytes;
        const int ic_byte_base = block * state->block_size + byte * 8;
        const __fp16* src = state->src_h + ((size_t)(ic_byte_base >> 6) * state->m + state->pos) * 64 +
                            (ic_byte_base & 63);
        __fp16* lut = state->table_lut_h + (size_t)k * 2 * 64;
        tmac_store_subset_lut_h(lut, src);
        tmac_store_subset_lut_h(lut + 64, src + 4);
    }
}

static inline void tmac_build_table_lut_h_range_oneblock(TmacBuildTableState* state, int start, int end) {
    const __fp16* src = state->src_h + (size_t)start * 8;
    __fp16* lut = state->table_lut_h + (size_t)start * 2 * 64;
    for (int k = start; k < end; ++k) {
        tmac_store_subset_lut_h(lut, src);
        tmac_store_subset_lut_h(lut + 64, src + 4);
        src += 8;
        lut += 128;
    }
}

static void tmac_build_table_worker(void* data, int worker_index) {
    (void)worker_index;
    TmacBuildTableTask* task = (TmacBuildTableTask*)data;
    if (task->state->scale_block_num == 1 && task->state->m == 1) {
        tmac_build_table_lut_h_range_oneblock(task->state, task->start, task->end);
    } else {
        tmac_build_table_lut_h_range(task->state, task->start, task->end);
    }
    worker_pool_synctoken_jobdone(&task->state->sync_ctx);
}

static inline void tmac_build_table_lut_h_range_m2(TmacBuildTableM2State* state, int start, int end) {
    for (int k = start; k < end; ++k) {
        const int block = k / state->weight_block_bytes;
        const int byte = k - block * state->weight_block_bytes;
        const int ic_byte_base = block * state->block_size + byte * 8;
        const __fp16* src0 = state->src_h + ((size_t)(ic_byte_base >> 6) * 2) * 64 + (ic_byte_base & 63);
        const __fp16* src1 = src0 + 64;
        __fp16* lut0 = state->table_lut_h0 + (size_t)k * 2 * 64;
        __fp16* lut1 = state->table_lut_h1 + (size_t)k * 2 * 64;
        tmac_store_subset_lut_h(lut0, src0);
        tmac_store_subset_lut_h(lut0 + 64, src0 + 4);
        tmac_store_subset_lut_h(lut1, src1);
        tmac_store_subset_lut_h(lut1 + 64, src1 + 4);
    }
}

static void tmac_build_table_m2_worker(void* data, int worker_index) {
    (void)worker_index;
    TmacBuildTableM2Task* task = (TmacBuildTableM2Task*)data;
    tmac_build_table_lut_h_range_m2(task->state, task->start, task->end);
    worker_pool_synctoken_jobdone(&task->state->sync_ctx);
}

static inline void tmac_build_table_lut_h_m2(__fp16* table_lut_h0, __fp16* table_lut_h1,
                                             const __fp16* src_h, int block_size,
                                             int weight_block_bytes) {
    TmacBuildTableM2State state = {table_lut_h0, table_lut_h1, src_h, block_size,
                                   weight_block_bytes, weight_block_bytes};
    int task_count = 1;
    if (weight_block_bytes >= 96 && g_max_num_workers > 1) {
        task_count = (int)g_max_num_workers;
        const int table_chunks = (weight_block_bytes + 47) / 48;
        if (task_count > table_chunks) {
            task_count = table_chunks;
        }
    }
    if (task_count <= 1) {
        tmac_build_table_lut_h_range_m2(&state, 0, weight_block_bytes);
        return;
    }
    TmacBuildTableM2Task* tasks = WORKER_POOL_STACK_ALLOC(TmacBuildTableM2Task, task_count);
    worker_pool_job_t job;
    job.fptr = tmac_build_table_m2_worker;
    worker_pool_synctoken_init(&state.sync_ctx, task_count);
    for (int t = 0; t < task_count; ++t) {
        tasks[t].state = &state;
        tasks[t].start = weight_block_bytes * t / task_count;
        tasks[t].end = weight_block_bytes * (t + 1) / task_count;
        job.dptr = tasks + t;
        worker_pool_submit(NULL, job);
    }
    worker_pool_synctoken_wait(&state.sync_ctx);
}

static inline void tmac_build_table_lut_h(__fp16* table_lut_h, const __fp16* src_h,
                                          int m, int pos, int block_size,
                                          int weight_block_bytes, int scale_block_num) {
    const int total_weight_bytes = scale_block_num * weight_block_bytes;
    TmacBuildTableState state = {table_lut_h, src_h, m, pos, block_size, weight_block_bytes,
                                 scale_block_num, total_weight_bytes};
    int task_count = 1;
    if (total_weight_bytes >= 256 && g_max_num_workers > 1) {
        task_count = (int)g_max_num_workers;
        const int table_chunks = (total_weight_bytes + 47) / 48;
        if (task_count > table_chunks) {
            task_count = table_chunks;
        }
    }
    if (task_count <= 1) {
        if (scale_block_num == 1 && m == 1) {
            tmac_build_table_lut_h_range_oneblock(&state, 0, total_weight_bytes);
        } else {
            tmac_build_table_lut_h_range(&state, 0, total_weight_bytes);
        }
    } else {
        TmacBuildTableTask* tasks = WORKER_POOL_STACK_ALLOC(TmacBuildTableTask, task_count);
        worker_pool_job_t job;
        job.fptr = tmac_build_table_worker;
        worker_pool_synctoken_init(&state.sync_ctx, task_count);
        for (int t = 0; t < task_count; ++t) {
            tasks[t].state = &state;
            tasks[t].start = total_weight_bytes * t / task_count;
            tasks[t].end = total_weight_bytes * (t + 1) / task_count;
            job.dptr = tasks + t;
            worker_pool_submit(NULL, job);
        }
        worker_pool_synctoken_wait(&state.sync_ctx);
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

static inline void tmac_vlut_accumulate_byte_indices(HVX_Vector* chunk0_h, HVX_Vector* chunk1_h,
                                                     HVX_Vector vWeightByte, HVX_Vector vHighIdx,
                                                     HVX_Vector vInvHighIdx, HVX_Vector vInvLowIdx,
                                                     const __fp16* table_high_h) {
    HVX_Vector vTableHigh = *((const HVX_Vector*)table_high_h);
    HVX_Vector vTableLow = *((const HVX_Vector*)(table_high_h + 64));
    HVX_VectorPair high_pair = Q6_Wh_vlut16_VbVhR_nomatch(vHighIdx, vTableHigh, 0);
    HVX_VectorPair low_pair = Q6_Wh_vlut16_VbVhR_nomatch(vWeightByte, vTableLow, 0);
    HVX_VectorPair inv_high_pair = Q6_Wh_vlut16_VbVhR_nomatch(vInvHighIdx, vTableHigh, 0);
    HVX_VectorPair inv_low_pair = Q6_Wh_vlut16_VbVhR_nomatch(vInvLowIdx, vTableLow, 0);
    HVX_Vector selected0_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_V_lo_W(high_pair),
                                                                      Q6_V_lo_W(low_pair)));
    HVX_Vector selected1_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_V_hi_W(high_pair),
                                                                      Q6_V_hi_W(low_pair)));
    HVX_Vector unselected0_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_V_lo_W(inv_high_pair),
                                                                        Q6_V_lo_W(inv_low_pair)));
    HVX_Vector unselected1_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(Q6_V_hi_W(inv_high_pair),
                                                                        Q6_V_hi_W(inv_low_pair)));
    HVX_Vector signed0_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(selected0_h, unselected0_h));
    HVX_Vector signed1_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(selected1_h, unselected1_h));
    *chunk0_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(*chunk0_h, signed0_h));
    *chunk1_h = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(*chunk1_h, signed1_h));
}

static inline void tmac_vlut_accumulate_byte(HVX_Vector* chunk0_h, HVX_Vector* chunk1_h,
                                             const uint8_t* weight_byte, const __fp16* table_high_h) {
    HVX_Vector vWeightByte = vmemu(weight_byte);
    HVX_Vector vHighIdx = Q6_Vub_vlsr_VubR(vWeightByte, 4);
    HVX_Vector vInvMask = Q6_V_vsplat_R(0x0f0f0f0f);
    tmac_vlut_accumulate_byte_indices(chunk0_h, chunk1_h, vWeightByte, vHighIdx,
                                      Q6_V_vxor_VV(vHighIdx, vInvMask),
                                      Q6_V_vxor_VV(vWeightByte, vInvMask), table_high_h);
}

static void compute_tmac_hvx_pack_range(TmacA16W1HvxState* state, int oc_pack_start, int oc_pack_end) {
    const int output_oc_pack = 64;
    const int weight_oc_pack = 128;

    for (int oc_pack_index = oc_pack_start; oc_pack_index < oc_pack_end; ++oc_pack_index) {
        const int oc_base = oc_pack_index * weight_oc_pack;
        const uint8_t* weight_pack = state->weight + (size_t)oc_pack_index *
                                     state->scale_block_num * state->weight_block_bytes * 128;
        l2fetch(weight_pack, 128, 128, state->weight_block_bytes, 0);
        HVX_Vector acc00 = Q6_V_vzero();
        HVX_Vector acc01 = Q6_V_vzero();
        HVX_Vector acc10 = Q6_V_vzero();
        HVX_Vector acc11 = Q6_V_vzero();
        if (state->bias != 0) {
            const float* bias_pack = state->bias + (size_t)oc_pack_index * 4 * 32;
            acc00 = vmemu(bias_pack + 0 * 32);
            acc01 = vmemu(bias_pack + 1 * 32);
            acc10 = vmemu(bias_pack + 2 * 32);
            acc11 = vmemu(bias_pack + 3 * 32);
        }

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
                HVX_Vector chunk0_h = Q6_V_vzero();
                HVX_Vector chunk1_h = Q6_V_vzero();
                for (int i = 0; i < 8; ++i) {
                    tmac_vlut_accumulate_byte(&chunk0_h, &chunk1_h, weight_byte, table_high_h);
                    weight_byte += 128;
                    table_high_h += 128;
                }
                tmac_accumulate_lookup_h_to_f32(&selected00, &selected01, &selected10, &selected11,
                                                chunk0_h, chunk1_h);
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

            const float* scale_pack = state->scale + ((size_t)oc_pack_index * state->scale_block_num + block) * 4 * 32;
            HVX_Vector scaled00 = Q6_Vsf_vmpy_VsfVsf(selected00, *((const HVX_Vector*)(scale_pack + 0 * 32)));
            HVX_Vector scaled01 = Q6_Vsf_vmpy_VsfVsf(selected01, *((const HVX_Vector*)(scale_pack + 1 * 32)));
            acc00 = (Q6_Vsf_vadd_VsfVsf(acc00, scaled00));
            acc01 = (Q6_Vsf_vadd_VsfVsf(acc01, scaled01));
            HVX_Vector scaled10 = Q6_Vsf_vmpy_VsfVsf(selected10, *((const HVX_Vector*)(scale_pack + 2 * 32)));
            HVX_Vector scaled11 = Q6_Vsf_vmpy_VsfVsf(selected11, *((const HVX_Vector*)(scale_pack + 3 * 32)));
            acc10 = (Q6_Vsf_vadd_VsfVsf(acc10, scaled10));
            acc11 = (Q6_Vsf_vadd_VsfVsf(acc11, scaled11));
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

static void compute_tmac_hvx_pack_range_oneblock(TmacA16W1HvxState* state, int oc_pack_start, int oc_pack_end) {
    const int output_oc_pack = 64;
    const int weight_oc_pack = 128;

    for (int oc_pack_index = oc_pack_start; oc_pack_index < oc_pack_end; ++oc_pack_index) {
        const int oc_base = oc_pack_index * weight_oc_pack;
        const uint8_t* weight_byte = state->weight + (size_t)oc_pack_index *
                                     state->weight_block_bytes * 128;
        l2fetch(weight_byte, 128, 128, state->weight_block_bytes, 0);
        HVX_Vector acc00 = Q6_V_vzero();
        HVX_Vector acc01 = Q6_V_vzero();
        HVX_Vector acc10 = Q6_V_vzero();
        HVX_Vector acc11 = Q6_V_vzero();
        if (state->bias != 0) {
            const float* bias_pack = state->bias + (size_t)oc_pack_index * 4 * 32;
            acc00 = vmemu(bias_pack + 0 * 32);
            acc01 = vmemu(bias_pack + 1 * 32);
            acc10 = vmemu(bias_pack + 2 * 32);
            acc11 = vmemu(bias_pack + 3 * 32);
        }

        HVX_Vector selected00 = Q6_V_vzero();
        HVX_Vector selected01 = Q6_V_vzero();
        HVX_Vector selected10 = Q6_V_vzero();
        HVX_Vector selected11 = Q6_V_vzero();
        const __fp16* table_high_h = state->table_lut_h;
        int byte = 0;
        for (; byte + 7 < state->weight_block_bytes; byte += 8) {
            HVX_Vector chunk0_h = Q6_V_vzero();
            HVX_Vector chunk1_h = Q6_V_vzero();
            for (int i = 0; i < 8; ++i) {
                tmac_vlut_accumulate_byte(&chunk0_h, &chunk1_h, weight_byte, table_high_h);
                weight_byte += 128;
                table_high_h += 128;
            }
            tmac_accumulate_lookup_h_to_f32(&selected00, &selected01, &selected10, &selected11,
                                            chunk0_h, chunk1_h);
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

        const float* scale_pack = state->scale + (size_t)oc_pack_index * 4 * 32;
        HVX_Vector scaled00 = Q6_Vsf_vmpy_VsfVsf(selected00, *((const HVX_Vector*)(scale_pack + 0 * 32)));
        HVX_Vector scaled01 = Q6_Vsf_vmpy_VsfVsf(selected01, *((const HVX_Vector*)(scale_pack + 1 * 32)));
        acc00 = Q6_Vsf_vadd_VsfVsf(acc00, scaled00);
        acc01 = Q6_Vsf_vadd_VsfVsf(acc01, scaled01);
        HVX_Vector scaled10 = Q6_Vsf_vmpy_VsfVsf(selected10, *((const HVX_Vector*)(scale_pack + 2 * 32)));
        HVX_Vector scaled11 = Q6_Vsf_vmpy_VsfVsf(selected11, *((const HVX_Vector*)(scale_pack + 3 * 32)));
        acc10 = Q6_Vsf_vadd_VsfVsf(acc10, scaled10);
        acc11 = Q6_Vsf_vadd_VsfVsf(acc11, scaled11);

        const int output_pack_index = oc_pack_index * 2;
        HVX_Vector acc0 = Q6_Vhf_vcvt_VsfVsf(acc00, acc01);
        vmemu(state->dst_h + ((size_t)output_pack_index * state->m + state->pos) * output_oc_pack) = acc0;
        if (oc_base + output_oc_pack < ((state->oc + output_oc_pack - 1) / output_oc_pack) * output_oc_pack) {
            HVX_Vector acc1 = Q6_Vhf_vcvt_VsfVsf(acc10, acc11);
            vmemu(state->dst_h + ((size_t)(output_pack_index + 1) * state->m + state->pos) * output_oc_pack) = acc1;
        }
    }
}

static void compute_tmac_hvx_pack_range_oneblock_m2(TmacA16W1HvxState* state, int oc_pack_start, int oc_pack_end) {
    const int output_oc_pack = 64;
    const int weight_oc_pack = 128;
    const __fp16* table_lut_h1 = state->table_lut_h + (size_t)state->weight_block_bytes * 2 * 64;
    const HVX_Vector vInvMask = Q6_V_vsplat_R(0x0f0f0f0f);

    for (int oc_pack_index = oc_pack_start; oc_pack_index < oc_pack_end; ++oc_pack_index) {
        const int oc_base = oc_pack_index * weight_oc_pack;
        const uint8_t* weight_byte_base = state->weight + (size_t)oc_pack_index *
                                          state->weight_block_bytes * 128;
        l2fetch(weight_byte_base, 128, 128, state->weight_block_bytes, 0);

        HVX_Vector acc00_0 = Q6_V_vzero();
        HVX_Vector acc01_0 = Q6_V_vzero();
        HVX_Vector acc10_0 = Q6_V_vzero();
        HVX_Vector acc11_0 = Q6_V_vzero();
        HVX_Vector acc00_1 = Q6_V_vzero();
        HVX_Vector acc01_1 = Q6_V_vzero();
        HVX_Vector acc10_1 = Q6_V_vzero();
        HVX_Vector acc11_1 = Q6_V_vzero();
        if (state->bias != 0) {
            const float* bias_pack = state->bias + (size_t)oc_pack_index * 4 * 32;
            acc00_0 = vmemu(bias_pack + 0 * 32);
            acc01_0 = vmemu(bias_pack + 1 * 32);
            acc10_0 = vmemu(bias_pack + 2 * 32);
            acc11_0 = vmemu(bias_pack + 3 * 32);
            acc00_1 = acc00_0;
            acc01_1 = acc01_0;
            acc10_1 = acc10_0;
            acc11_1 = acc11_0;
        }

        HVX_Vector selected00_0 = Q6_V_vzero();
        HVX_Vector selected01_0 = Q6_V_vzero();
        HVX_Vector selected10_0 = Q6_V_vzero();
        HVX_Vector selected11_0 = Q6_V_vzero();
        HVX_Vector selected00_1 = Q6_V_vzero();
        HVX_Vector selected01_1 = Q6_V_vzero();
        HVX_Vector selected10_1 = Q6_V_vzero();
        HVX_Vector selected11_1 = Q6_V_vzero();

        const uint8_t* weight_byte = weight_byte_base;
        const __fp16* table0 = state->table_lut_h;
        const __fp16* table1 = table_lut_h1;
        int byte = 0;
        for (; byte + 7 < state->weight_block_bytes; byte += 8) {
            HVX_Vector chunk0_h0 = Q6_V_vzero();
            HVX_Vector chunk1_h0 = Q6_V_vzero();
            HVX_Vector chunk0_h1 = Q6_V_vzero();
            HVX_Vector chunk1_h1 = Q6_V_vzero();
            for (int i = 0; i < 8; ++i) {
                HVX_Vector vWeightByte = vmemu(weight_byte);
                HVX_Vector vHighIdx = Q6_Vub_vlsr_VubR(vWeightByte, 4);
                HVX_Vector vInvHighIdx = Q6_V_vxor_VV(vHighIdx, vInvMask);
                HVX_Vector vInvLowIdx = Q6_V_vxor_VV(vWeightByte, vInvMask);
                tmac_vlut_accumulate_byte_indices(&chunk0_h0, &chunk1_h0, vWeightByte, vHighIdx,
                                                  vInvHighIdx, vInvLowIdx, table0);
                tmac_vlut_accumulate_byte_indices(&chunk0_h1, &chunk1_h1, vWeightByte, vHighIdx,
                                                  vInvHighIdx, vInvLowIdx, table1);
                weight_byte += 128;
                table0 += 128;
                table1 += 128;
            }
            tmac_accumulate_lookup_h_to_f32(&selected00_0, &selected01_0, &selected10_0, &selected11_0,
                                            chunk0_h0, chunk1_h0);
            tmac_accumulate_lookup_h_to_f32(&selected00_1, &selected01_1, &selected10_1, &selected11_1,
                                            chunk0_h1, chunk1_h1);
        }
        if (byte < state->weight_block_bytes) {
            HVX_Vector chunk0_h0 = Q6_V_vzero();
            HVX_Vector chunk1_h0 = Q6_V_vzero();
            HVX_Vector chunk0_h1 = Q6_V_vzero();
            HVX_Vector chunk1_h1 = Q6_V_vzero();
            for (; byte < state->weight_block_bytes; ++byte) {
                HVX_Vector vWeightByte = vmemu(weight_byte);
                HVX_Vector vHighIdx = Q6_Vub_vlsr_VubR(vWeightByte, 4);
                HVX_Vector vInvHighIdx = Q6_V_vxor_VV(vHighIdx, vInvMask);
                HVX_Vector vInvLowIdx = Q6_V_vxor_VV(vWeightByte, vInvMask);
                tmac_vlut_accumulate_byte_indices(&chunk0_h0, &chunk1_h0, vWeightByte, vHighIdx,
                                                  vInvHighIdx, vInvLowIdx, table0);
                tmac_vlut_accumulate_byte_indices(&chunk0_h1, &chunk1_h1, vWeightByte, vHighIdx,
                                                  vInvHighIdx, vInvLowIdx, table1);
                weight_byte += 128;
                table0 += 128;
                table1 += 128;
            }
            tmac_accumulate_lookup_h_to_f32(&selected00_0, &selected01_0, &selected10_0, &selected11_0,
                                            chunk0_h0, chunk1_h0);
            tmac_accumulate_lookup_h_to_f32(&selected00_1, &selected01_1, &selected10_1, &selected11_1,
                                            chunk0_h1, chunk1_h1);
        }

        const float* scale_pack = state->scale + (size_t)oc_pack_index * 4 * 32;
        HVX_Vector scale00 = *((const HVX_Vector*)(scale_pack + 0 * 32));
        HVX_Vector scale01 = *((const HVX_Vector*)(scale_pack + 1 * 32));
        HVX_Vector scale10 = *((const HVX_Vector*)(scale_pack + 2 * 32));
        HVX_Vector scale11 = *((const HVX_Vector*)(scale_pack + 3 * 32));
        acc00_0 = Q6_Vsf_vadd_VsfVsf(acc00_0, Q6_Vsf_vmpy_VsfVsf(selected00_0, scale00));
        acc01_0 = Q6_Vsf_vadd_VsfVsf(acc01_0, Q6_Vsf_vmpy_VsfVsf(selected01_0, scale01));
        acc10_0 = Q6_Vsf_vadd_VsfVsf(acc10_0, Q6_Vsf_vmpy_VsfVsf(selected10_0, scale10));
        acc11_0 = Q6_Vsf_vadd_VsfVsf(acc11_0, Q6_Vsf_vmpy_VsfVsf(selected11_0, scale11));
        acc00_1 = Q6_Vsf_vadd_VsfVsf(acc00_1, Q6_Vsf_vmpy_VsfVsf(selected00_1, scale00));
        acc01_1 = Q6_Vsf_vadd_VsfVsf(acc01_1, Q6_Vsf_vmpy_VsfVsf(selected01_1, scale01));
        acc10_1 = Q6_Vsf_vadd_VsfVsf(acc10_1, Q6_Vsf_vmpy_VsfVsf(selected10_1, scale10));
        acc11_1 = Q6_Vsf_vadd_VsfVsf(acc11_1, Q6_Vsf_vmpy_VsfVsf(selected11_1, scale11));

        const int output_pack_index = oc_pack_index * 2;
        HVX_Vector acc0_0 = Q6_Vhf_vcvt_VsfVsf(acc00_0, acc01_0);
        HVX_Vector acc0_1 = Q6_Vhf_vcvt_VsfVsf(acc00_1, acc01_1);
        vmemu(state->dst_h + ((size_t)output_pack_index * state->m + 0) * output_oc_pack) = acc0_0;
        vmemu(state->dst_h + ((size_t)output_pack_index * state->m + 1) * output_oc_pack) = acc0_1;
        if (oc_base + output_oc_pack < ((state->oc + output_oc_pack - 1) / output_oc_pack) * output_oc_pack) {
            HVX_Vector acc1_0 = Q6_Vhf_vcvt_VsfVsf(acc10_0, acc11_0);
            HVX_Vector acc1_1 = Q6_Vhf_vcvt_VsfVsf(acc10_1, acc11_1);
            vmemu(state->dst_h + ((size_t)(output_pack_index + 1) * state->m + 0) * output_oc_pack) = acc1_0;
            vmemu(state->dst_h + ((size_t)(output_pack_index + 1) * state->m + 1) * output_oc_pack) = acc1_1;
        }
    }
}

static void compute_tmac_hvx_worker(void* data, int worker_index) {
    (void)worker_index;
    TmacA16W1HvxTask* task = (TmacA16W1HvxTask*)data;
    if (task->state->scale_block_num == 1) {
        compute_tmac_hvx_pack_range_oneblock(task->state, task->oc_pack_start, task->oc_pack_end);
    } else {
        compute_tmac_hvx_pack_range(task->state, task->oc_pack_start, task->oc_pack_end);
    }
    worker_pool_synctoken_jobdone(&task->state->sync_ctx);
}

static void compute_tmac_hvx_m2_oneblock_worker(void* data, int worker_index) {
    (void)worker_index;
    TmacA16W1HvxTask* task = (TmacA16W1HvxTask*)data;
    compute_tmac_hvx_pack_range_oneblock_m2(task->state, task->oc_pack_start, task->oc_pack_end);
    worker_pool_synctoken_jobdone(&task->state->sync_ctx);
}

static int compute_tmac_hvx_m2_oneblock(uint8_t *dst, const uint8_t *src, const uint8_t *weight,
                                        const float *scale, const uint8_t *bias,
                                        int ic, int oc, int output_bytes) {
    const int output_oc_pack = 64;
    const int weight_oc_pack = 128;
    const int oc_round = ((oc + output_oc_pack - 1) / output_oc_pack) * output_oc_pack;
    const int output_oc_pack_count = oc_round / output_oc_pack;
    const int weight_oc_pack_count = (oc + weight_oc_pack - 1) / weight_oc_pack;
    const int weight_block_bytes = ic >> 3;
    const int needed_output_bytes = output_oc_pack_count * 2 * output_oc_pack * (int)sizeof(__fp16);
    if (output_bytes > 0 && needed_output_bytes > output_bytes) {
        return AEE_EBADPARM;
    }

    const __fp16* src_h = (const __fp16*)src;
    __fp16* dst_h = (__fp16*)dst;
    const float* bias_pack = (const float*)bias;
    const size_t table_lut_count = (size_t)weight_block_bytes * 2 * 64;
    uint8_t* vtcm_ptr = (uint8_t*)vtcm_manager_get_vtcm_base();
    if (vtcm_ptr == 0) {
        return AEE_EUNSUPPORTED;
    }
    __fp16* table_lut_h = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, table_lut_count * 2 * sizeof(__fp16));
    __fp16* table_lut_h1 = table_lut_h + table_lut_count;
    tmac_build_table_lut_h_m2(table_lut_h, table_lut_h1, src_h, ic, weight_block_bytes);

    int task_count = 1;
    if (weight_oc_pack_count >= 2 && g_max_num_workers > 1) {
        task_count = (int)g_max_num_workers;
        if (task_count > weight_oc_pack_count) {
            task_count = weight_oc_pack_count;
        }
    }
    TmacA16W1HvxState state = {dst_h, src_h, weight, scale, bias_pack, 2, 0, ic, oc,
                               1, ic, weight_block_bytes, table_lut_h};
    if (task_count <= 1) {
        TmacA16W1HvxTask task = {&state, 0, weight_oc_pack_count};
        compute_tmac_hvx_m2_oneblock_worker(&task, 0);
        return AEE_SUCCESS;
    }
    TmacA16W1HvxTask* tasks = WORKER_POOL_STACK_ALLOC(TmacA16W1HvxTask, task_count);
    worker_pool_job_t job;
    job.fptr = compute_tmac_hvx_m2_oneblock_worker;
    worker_pool_synctoken_init(&state.sync_ctx, task_count);
    for (int t = 0; t < task_count; ++t) {
        tasks[t].state = &state;
        tasks[t].oc_pack_start = weight_oc_pack_count * t / task_count;
        tasks[t].oc_pack_end = weight_oc_pack_count * (t + 1) / task_count;
        job.dptr = tasks + t;
        worker_pool_submit(NULL, job);
    }
    worker_pool_synctoken_wait(&state.sync_ctx);
    return AEE_SUCCESS;
}

static int compute_tmac_hvx(uint8_t *dst, const uint8_t *src, const uint8_t *weight, const float *scale,
                            const uint8_t *bias, int m, int ic, int oc, int scale_block_num,
                            int output_bytes) {
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
    if (m == 2 && scale_block_num == 1) {
        return compute_tmac_hvx_m2_oneblock(dst, src, weight, scale, bias, ic, oc, output_bytes);
    }

    const __fp16* src_h = (const __fp16*)src;
    __fp16* dst_h = (__fp16*)dst;
    const float* bias_pack = (const float*)bias;
    const size_t table_lut_count = (size_t)scale_block_num * weight_block_bytes * 2 * 64;
    uint8_t* vtcm_ptr = (uint8_t*)vtcm_manager_get_vtcm_base();
    if (vtcm_ptr == 0) {
        return AEE_EUNSUPPORTED;
    }
    __fp16* table_lut_h = (__fp16*)vtcm_seq_alloc(&vtcm_ptr, table_lut_count * sizeof(__fp16));

    for (int pos = 0; pos < m; ++pos) {
        tmac_build_table_lut_h(table_lut_h, src_h, m, pos, block_size, weight_block_bytes, scale_block_num);
        int task_count = 1;
        if (weight_oc_pack_count >= 2 && g_max_num_workers > 1) {
            task_count = (int)g_max_num_workers;
            if (task_count > weight_oc_pack_count) {
                task_count = weight_oc_pack_count;
            }
        }
        TmacA16W1HvxState state = {dst_h, src_h, weight, scale, bias_pack, m, pos, ic, oc,
                                   scale_block_num, block_size, weight_block_bytes,
                                   table_lut_h};
        if (task_count <= 1) {
            if (scale_block_num == 1) {
                compute_tmac_hvx_pack_range_oneblock(&state, 0, weight_oc_pack_count);
            } else {
                compute_tmac_hvx_pack_range(&state, 0, weight_oc_pack_count);
            }
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

int hvx_tmac_a16w1_fp16(uint8_t *dst, const uint8_t *src, const uint8_t *weight, const float *scale,
                        const uint8_t *bias, int m, int ic, int oc, int scale_block_num,
                        int scale_asymmetric, int output_bytes) {
    (void)scale_asymmetric;
    if (dst == 0 || src == 0 || weight == 0 || scale == 0 || m <= 0 || ic <= 0 || oc <= 0 ||
        scale_block_num <= 0 || (ic % scale_block_num) != 0) {
        return AEE_EBADPARM;
    }
    const int block_size = ic / scale_block_num;
    if ((block_size & 7) != 0) {
        return AEE_EBADPARM;
    }
    return compute_tmac_hvx(dst, src, weight, scale, bias, m, ic, oc, scale_block_num,
                            output_bytes);
}

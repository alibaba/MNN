#include <riscv_vector.h>
#include <stddef.h>

// Match the RISC-V scalar fallback's explicit FMA contract: round the rank-one
// product before fusing decay, and fuse v - decay * outK before multiplying beta.

void MNNRankOneUpdate_RVV(float* S, const float* k, const float* delta, size_t dk, size_t dv) {
    for (size_t i = 0; i < dk; ++i) {
        const float kValue = k[i];
        float* row = S + i * dv;
        size_t j = 0;
        while (j < dv) {
            const size_t vl = __riscv_vsetvl_e32m8(dv - j);
            vfloat32m8_t state = __riscv_vle32_v_f32m8(row + j, vl);
            const vfloat32m8_t deltaValue = __riscv_vle32_v_f32m8(delta + j, vl);
            state = __riscv_vfmacc_vf_f32m8(state, kValue, deltaValue, vl);
            __riscv_vse32_v_f32m8(row + j, state, vl);
            j += vl;
        }
    }
}

void MNNDualMatVec_RVV(const float* S, const float* k, const float* q, float* out_k, float* out_q, size_t dk,
                       size_t dv) {
    size_t j = 0;
    while (j < dv) {
        const size_t vl = __riscv_vsetvl_e32m8(dv - j);
        __riscv_vse32_v_f32m8(out_k + j, __riscv_vfmv_v_f_f32m8(0.0f, vl), vl);
        __riscv_vse32_v_f32m8(out_q + j, __riscv_vfmv_v_f_f32m8(0.0f, vl), vl);
        j += vl;
    }

    for (size_t i = 0; i < dk; ++i) {
        const float kValue = k[i];
        const float qValue = q[i];
        const float* row = S + i * dv;
        j = 0;
        while (j < dv) {
            const size_t vl = __riscv_vsetvl_e32m8(dv - j);
            const vfloat32m8_t state = __riscv_vle32_v_f32m8(row + j, vl);
            vfloat32m8_t outK = __riscv_vle32_v_f32m8(out_k + j, vl);
            vfloat32m8_t outQ = __riscv_vle32_v_f32m8(out_q + j, vl);
            outK = __riscv_vfmacc_vf_f32m8(outK, kValue, state, vl);
            outQ = __riscv_vfmacc_vf_f32m8(outQ, qValue, state, vl);
            __riscv_vse32_v_f32m8(out_k + j, outK, vl);
            __riscv_vse32_v_f32m8(out_q + j, outQ, vl);
            j += vl;
        }
    }
}

void MNNDecayRankOneUpdate_RVV(float* S, const float* k, const float* delta, float decay, size_t dk, size_t dv) {
    for (size_t i = 0; i < dk; ++i) {
        const float kValue = k[i];
        float* row = S + i * dv;
        size_t j = 0;
        while (j < dv) {
            const size_t vl = __riscv_vsetvl_e32m8(dv - j);
            vfloat32m8_t state = __riscv_vle32_v_f32m8(row + j, vl);
            const vfloat32m8_t deltaValue = __riscv_vle32_v_f32m8(delta + j, vl);
            const vfloat32m8_t update = __riscv_vfmul_vf_f32m8(deltaValue, kValue, vl);
            state = __riscv_vfmacc_vf_f32m8(update, decay, state, vl);
            __riscv_vse32_v_f32m8(row + j, state, vl);
            j += vl;
        }
    }
}

void MNNFusedGatedDelta_RVV(float* S, const float* k, const float* q, const float* v, float* out, float decay,
                            float beta, float kq, size_t dk, size_t dv) {
    size_t j = 0;
    while (j < dv) {
        const size_t vl = __riscv_vsetvl_e32m8(dv - j);
        vfloat32m8_t outK = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        vfloat32m8_t outQ = __riscv_vfmv_v_f_f32m8(0.0f, vl);

        for (size_t i = 0; i < dk; ++i) {
            const vfloat32m8_t state = __riscv_vle32_v_f32m8(S + i * dv + j, vl);
            outK = __riscv_vfmacc_vf_f32m8(outK, k[i], state, vl);
            outQ = __riscv_vfmacc_vf_f32m8(outQ, q[i], state, vl);
        }

        const vfloat32m8_t value = __riscv_vle32_v_f32m8(v + j, vl);
        vfloat32m8_t delta = __riscv_vfnmsac_vf_f32m8(value, decay, outK, vl);
        delta = __riscv_vfmul_vf_f32m8(delta, beta, vl);

        vfloat32m8_t outValue = __riscv_vfmul_vf_f32m8(delta, kq, vl);
        outValue = __riscv_vfmacc_vf_f32m8(outValue, decay, outQ, vl);
        __riscv_vse32_v_f32m8(out + j, outValue, vl);

        for (size_t i = 0; i < dk; ++i) {
            float* row = S + i * dv + j;
            vfloat32m8_t state = __riscv_vle32_v_f32m8(row, vl);
            const vfloat32m8_t update = __riscv_vfmul_vf_f32m8(delta, k[i], vl);
            state = __riscv_vfmacc_vf_f32m8(update, decay, state, vl);
            __riscv_vse32_v_f32m8(row, state, vl);
        }
        j += vl;
    }
}

//
//  LinearAttentionFunctions.cpp
//  MNN
//
//  AVX-512 kernels for the gated-delta-rule recurrence used by LinearAttention.
//

#include "FunctionSummary.hpp"

namespace {

void _AVX512_MNNDualMatVec(const float* S, const float* k, const float* q, float* outK, float* outQ, size_t dk,
                           size_t dv) {
    size_t j = 0;
    for (; j + 16 <= dv; j += 16) {
        __m512 sumK = _mm512_setzero_ps();
        __m512 sumQ = _mm512_setzero_ps();
        for (size_t i = 0; i < dk; ++i) {
            const __m512 state = _mm512_loadu_ps(S + i * dv + j);
            sumK = _mm512_fmadd_ps(state, _mm512_set1_ps(k[i]), sumK);
            sumQ = _mm512_fmadd_ps(state, _mm512_set1_ps(q[i]), sumQ);
        }
        _mm512_storeu_ps(outK + j, sumK);
        _mm512_storeu_ps(outQ + j, sumQ);
    }
    for (; j < dv; ++j) {
        float sumK = 0.0f;
        float sumQ = 0.0f;
        for (size_t i = 0; i < dk; ++i) {
            const float state = S[i * dv + j];
            sumK += state * k[i];
            sumQ += state * q[i];
        }
        outK[j] = sumK;
        outQ[j] = sumQ;
    }
}

float _AVX512_MNNNormalizeQKAndDot(float* q, float* k, float qScale, bool useL2Norm, size_t dk) {
    __m512 qSum = _mm512_setzero_ps();
    __m512 kSum = _mm512_setzero_ps();
    __m512 qkSum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= dk; i += 16) {
        const __m512 qVec = _mm512_loadu_ps(q + i);
        const __m512 kVec = _mm512_loadu_ps(k + i);
        if (useL2Norm) {
            qSum = _mm512_fmadd_ps(qVec, qVec, qSum);
            kSum = _mm512_fmadd_ps(kVec, kVec, kSum);
        }
        qkSum = _mm512_fmadd_ps(qVec, kVec, qkSum);
    }
    alignas(64) float qs[16];
    alignas(64) float ks[16];
    alignas(64) float qks[16];
    _mm512_store_ps(qs, qSum);
    _mm512_store_ps(ks, kSum);
    _mm512_store_ps(qks, qkSum);
    float qSumScalar = 0.0f;
    float kSumScalar = 0.0f;
    float qkScalar = 0.0f;
    for (int lane = 0; lane < 16; ++lane) {
        qSumScalar += qs[lane];
        kSumScalar += ks[lane];
        qkScalar += qks[lane];
    }
    for (; i < dk; ++i) {
        if (useL2Norm) {
            qSumScalar += q[i] * q[i];
            kSumScalar += k[i] * k[i];
        }
        qkScalar += q[i] * k[i];
    }
    const float qNormScale = useL2Norm ? qScale / sqrtf(qSumScalar + 1e-6f) : qScale;
    const float kNormScale = useL2Norm ? 1.0f / sqrtf(kSumScalar + 1e-6f) : 1.0f;
    const __m512 qScaleVec = _mm512_set1_ps(qNormScale);
    const __m512 kScaleVec = _mm512_set1_ps(kNormScale);
    for (i = 0; i + 16 <= dk; i += 16) {
        _mm512_storeu_ps(q + i, _mm512_mul_ps(_mm512_loadu_ps(q + i), qScaleVec));
        if (useL2Norm) {
            _mm512_storeu_ps(k + i, _mm512_mul_ps(_mm512_loadu_ps(k + i), kScaleVec));
        }
    }
    for (; i < dk; ++i) {
        q[i] *= qNormScale;
        if (useL2Norm) {
            k[i] *= kNormScale;
        }
    }
    return qkScalar * qNormScale * kNormScale;
}

void _AVX512_MNNDecayRankOneUpdate(float* S, const float* k, const float* delta, float decay, size_t dk, size_t dv) {
    const __m512 decayVec = _mm512_set1_ps(decay);
    for (size_t i = 0; i < dk; ++i) {
        float* row = S + i * dv;
        const __m512 key = _mm512_set1_ps(k[i]);
        size_t j = 0;
        for (; j + 16 <= dv; j += 16) {
            const __m512 oldState = _mm512_loadu_ps(row + j);
            const __m512 deltaVec = _mm512_loadu_ps(delta + j);
            _mm512_storeu_ps(row + j, _mm512_fmadd_ps(key, deltaVec, _mm512_mul_ps(decayVec, oldState)));
        }
        for (; j < dv; ++j) {
            row[j] = decay * row[j] + k[i] * delta[j];
        }
    }
}

void _AVX512_MNNFusedGatedDelta(float* S, const float* k, const float* q, const float* v, float* out, float decay,
                                float beta, float kq, size_t dk, size_t dv) {
    const __m512 decayVec = _mm512_set1_ps(decay);
    const __m512 betaVec = _mm512_set1_ps(beta);
    const __m512 kqVec = _mm512_set1_ps(kq);
    size_t j = 0;
    // Process one 64-float state row at a time for common d_v=64/128 shapes.
    // This avoids re-walking the same rows once per 16-float vector block.
    for (; j + 64 <= dv; j += 64) {
        __m512 sumK0 = _mm512_setzero_ps();
        __m512 sumQ0 = _mm512_setzero_ps();
        __m512 sumK1 = _mm512_setzero_ps();
        __m512 sumQ1 = _mm512_setzero_ps();
        __m512 sumK2 = _mm512_setzero_ps();
        __m512 sumQ2 = _mm512_setzero_ps();
        __m512 sumK3 = _mm512_setzero_ps();
        __m512 sumQ3 = _mm512_setzero_ps();
        for (size_t i = 0; i < dk; ++i) {
            const float* row = S + i * dv + j;
            const __m512 key = _mm512_set1_ps(k[i]);
            const __m512 query = _mm512_set1_ps(q[i]);
            const __m512 state0 = _mm512_loadu_ps(row);
            const __m512 state1 = _mm512_loadu_ps(row + 16);
            const __m512 state2 = _mm512_loadu_ps(row + 32);
            const __m512 state3 = _mm512_loadu_ps(row + 48);
            sumK0 = _mm512_fmadd_ps(state0, key, sumK0);
            sumQ0 = _mm512_fmadd_ps(state0, query, sumQ0);
            sumK1 = _mm512_fmadd_ps(state1, key, sumK1);
            sumQ1 = _mm512_fmadd_ps(state1, query, sumQ1);
            sumK2 = _mm512_fmadd_ps(state2, key, sumK2);
            sumQ2 = _mm512_fmadd_ps(state2, query, sumQ2);
            sumK3 = _mm512_fmadd_ps(state3, key, sumK3);
            sumQ3 = _mm512_fmadd_ps(state3, query, sumQ3);
        }
        const __m512 delta0 =
            _mm512_mul_ps(betaVec, _mm512_sub_ps(_mm512_loadu_ps(v + j), _mm512_mul_ps(decayVec, sumK0)));
        const __m512 delta1 =
            _mm512_mul_ps(betaVec, _mm512_sub_ps(_mm512_loadu_ps(v + j + 16), _mm512_mul_ps(decayVec, sumK1)));
        const __m512 delta2 =
            _mm512_mul_ps(betaVec, _mm512_sub_ps(_mm512_loadu_ps(v + j + 32), _mm512_mul_ps(decayVec, sumK2)));
        const __m512 delta3 =
            _mm512_mul_ps(betaVec, _mm512_sub_ps(_mm512_loadu_ps(v + j + 48), _mm512_mul_ps(decayVec, sumK3)));
        _mm512_storeu_ps(out + j, _mm512_fmadd_ps(kqVec, delta0, _mm512_mul_ps(decayVec, sumQ0)));
        _mm512_storeu_ps(out + j + 16, _mm512_fmadd_ps(kqVec, delta1, _mm512_mul_ps(decayVec, sumQ1)));
        _mm512_storeu_ps(out + j + 32, _mm512_fmadd_ps(kqVec, delta2, _mm512_mul_ps(decayVec, sumQ2)));
        _mm512_storeu_ps(out + j + 48, _mm512_fmadd_ps(kqVec, delta3, _mm512_mul_ps(decayVec, sumQ3)));
        for (size_t i = 0; i < dk; ++i) {
            float* row = S + i * dv + j;
            const __m512 key = _mm512_set1_ps(k[i]);
            _mm512_storeu_ps(row, _mm512_fmadd_ps(key, delta0, _mm512_mul_ps(decayVec, _mm512_loadu_ps(row))));
            _mm512_storeu_ps(row + 16,
                             _mm512_fmadd_ps(key, delta1, _mm512_mul_ps(decayVec, _mm512_loadu_ps(row + 16))));
            _mm512_storeu_ps(row + 32,
                             _mm512_fmadd_ps(key, delta2, _mm512_mul_ps(decayVec, _mm512_loadu_ps(row + 32))));
            _mm512_storeu_ps(row + 48,
                             _mm512_fmadd_ps(key, delta3, _mm512_mul_ps(decayVec, _mm512_loadu_ps(row + 48))));
        }
    }
    for (; j + 16 <= dv; j += 16) {
        __m512 sumK = _mm512_setzero_ps();
        __m512 sumQ = _mm512_setzero_ps();
        for (size_t i = 0; i < dk; ++i) {
            const __m512 state = _mm512_loadu_ps(S + i * dv + j);
            sumK = _mm512_fmadd_ps(state, _mm512_set1_ps(k[i]), sumK);
            sumQ = _mm512_fmadd_ps(state, _mm512_set1_ps(q[i]), sumQ);
        }
        const __m512 delta =
            _mm512_mul_ps(betaVec, _mm512_sub_ps(_mm512_loadu_ps(v + j), _mm512_mul_ps(decayVec, sumK)));
        _mm512_storeu_ps(out + j, _mm512_fmadd_ps(kqVec, delta, _mm512_mul_ps(decayVec, sumQ)));
        for (size_t i = 0; i < dk; ++i) {
            float* row = S + i * dv + j;
            const __m512 state = _mm512_loadu_ps(row);
            _mm512_storeu_ps(row, _mm512_fmadd_ps(_mm512_set1_ps(k[i]), delta, _mm512_mul_ps(decayVec, state)));
        }
    }
    for (; j < dv; ++j) {
        float sumK = 0.0f;
        float sumQ = 0.0f;
        for (size_t i = 0; i < dk; ++i) {
            const float state = S[i * dv + j];
            sumK += state * k[i];
            sumQ += state * q[i];
        }
        const float delta = beta * (v[j] - decay * sumK);
        out[j] = decay * sumQ + kq * delta;
        for (size_t i = 0; i < dk; ++i) {
            S[i * dv + j] = decay * S[i * dv + j] + k[i] * delta;
        }
    }
}

} // namespace

extern "C" void _AVX512_LinearAttentionInit(void* functions) {
    auto coreFunction = static_cast<MNN::CoreFunctions*>(functions);
    coreFunction->MNNDualMatVec = _AVX512_MNNDualMatVec;
    coreFunction->MNNDecayRankOneUpdate = _AVX512_MNNDecayRankOneUpdate;
    coreFunction->MNNFusedGatedDelta = _AVX512_MNNFusedGatedDelta;
    coreFunction->MNNNormalizeQKAndDot = _AVX512_MNNNormalizeQKAndDot;
}

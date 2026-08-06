//
//  LinearAttentionFunctions.cpp
//  MNN
//

#include "FunctionSummary.hpp"

namespace {

using namespace MNN;

void _AVX_MNNDualMatVec(const float* S, const float* k, const float* q, float* outK, float* outQ, size_t dk,
                        size_t dv) {
    size_t j = 0;
    for (; j + 8 <= dv; j += 8) {
        __m256 sumK = _mm256_setzero_ps();
        __m256 sumQ = _mm256_setzero_ps();
        for (size_t i = 0; i < dk; ++i) {
            const __m256 state = _mm256_loadu_ps(S + i * dv + j);
            sumK = _mm256_add_ps(sumK, _mm256_mul_ps(state, _mm256_set1_ps(k[i])));
            sumQ = _mm256_add_ps(sumQ, _mm256_mul_ps(state, _mm256_set1_ps(q[i])));
        }
        _mm256_storeu_ps(outK + j, sumK);
        _mm256_storeu_ps(outQ + j, sumQ);
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

void _AVX_MNNDecayRankOneUpdate(float* S, const float* k, const float* delta, float decay, size_t dk, size_t dv) {
    const __m256 decayVec = _mm256_set1_ps(decay);
    for (size_t i = 0; i < dk; ++i) {
        float* row = S + i * dv;
        const __m256 key = _mm256_set1_ps(k[i]);
        size_t j = 0;
        for (; j + 8 <= dv; j += 8) {
            const __m256 state = _mm256_loadu_ps(row + j);
            const __m256 update = _mm256_mul_ps(key, _mm256_loadu_ps(delta + j));
            _mm256_storeu_ps(row + j, _mm256_add_ps(_mm256_mul_ps(decayVec, state), update));
        }
        for (; j < dv; ++j) {
            row[j] = decay * row[j] + k[i] * delta[j];
        }
    }
}

void _AVX_MNNFusedGatedDelta(float* S, const float* k, const float* q, const float* v, float* out, float decay,
                             float beta, float kq, size_t dk, size_t dv) {
    const __m256 decayVec = _mm256_set1_ps(decay);
    const __m256 betaVec = _mm256_set1_ps(beta);
    const __m256 kqVec = _mm256_set1_ps(kq);
    size_t j = 0;
    // Process two vectors per state row to improve locality for common
    // d_v=64/128 shapes without exhausting AVX2 registers.
    for (; j + 16 <= dv; j += 16) {
        __m256 sumK0 = _mm256_setzero_ps();
        __m256 sumQ0 = _mm256_setzero_ps();
        __m256 sumK1 = _mm256_setzero_ps();
        __m256 sumQ1 = _mm256_setzero_ps();
        for (size_t i = 0; i < dk; ++i) {
            const float* row = S + i * dv + j;
            const __m256 key = _mm256_set1_ps(k[i]);
            const __m256 query = _mm256_set1_ps(q[i]);
            const __m256 state0 = _mm256_loadu_ps(row);
            const __m256 state1 = _mm256_loadu_ps(row + 8);
            sumK0 = _mm256_add_ps(sumK0, _mm256_mul_ps(state0, key));
            sumQ0 = _mm256_add_ps(sumQ0, _mm256_mul_ps(state0, query));
            sumK1 = _mm256_add_ps(sumK1, _mm256_mul_ps(state1, key));
            sumQ1 = _mm256_add_ps(sumQ1, _mm256_mul_ps(state1, query));
        }
        const __m256 delta0 =
            _mm256_mul_ps(betaVec, _mm256_sub_ps(_mm256_loadu_ps(v + j), _mm256_mul_ps(decayVec, sumK0)));
        const __m256 delta1 =
            _mm256_mul_ps(betaVec, _mm256_sub_ps(_mm256_loadu_ps(v + j + 8), _mm256_mul_ps(decayVec, sumK1)));
        _mm256_storeu_ps(out + j, _mm256_add_ps(_mm256_mul_ps(decayVec, sumQ0), _mm256_mul_ps(kqVec, delta0)));
        _mm256_storeu_ps(out + j + 8, _mm256_add_ps(_mm256_mul_ps(decayVec, sumQ1), _mm256_mul_ps(kqVec, delta1)));
        for (size_t i = 0; i < dk; ++i) {
            float* row = S + i * dv + j;
            const __m256 key = _mm256_set1_ps(k[i]);
            _mm256_storeu_ps(row,
                             _mm256_add_ps(_mm256_mul_ps(decayVec, _mm256_loadu_ps(row)), _mm256_mul_ps(key, delta0)));
            _mm256_storeu_ps(
                row + 8, _mm256_add_ps(_mm256_mul_ps(decayVec, _mm256_loadu_ps(row + 8)), _mm256_mul_ps(key, delta1)));
        }
    }
    for (; j + 8 <= dv; j += 8) {
        __m256 sumK = _mm256_setzero_ps();
        __m256 sumQ = _mm256_setzero_ps();
        for (size_t i = 0; i < dk; ++i) {
            const __m256 state = _mm256_loadu_ps(S + i * dv + j);
            sumK = _mm256_add_ps(sumK, _mm256_mul_ps(state, _mm256_set1_ps(k[i])));
            sumQ = _mm256_add_ps(sumQ, _mm256_mul_ps(state, _mm256_set1_ps(q[i])));
        }
        const __m256 delta =
            _mm256_mul_ps(betaVec, _mm256_sub_ps(_mm256_loadu_ps(v + j), _mm256_mul_ps(decayVec, sumK)));
        _mm256_storeu_ps(out + j, _mm256_add_ps(_mm256_mul_ps(decayVec, sumQ), _mm256_mul_ps(kqVec, delta)));
        for (size_t i = 0; i < dk; ++i) {
            float* row = S + i * dv + j;
            const __m256 state = _mm256_loadu_ps(row);
            const __m256 update = _mm256_mul_ps(_mm256_set1_ps(k[i]), delta);
            _mm256_storeu_ps(row, _mm256_add_ps(_mm256_mul_ps(decayVec, state), update));
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

float _AVX_MNNNormalizeQKAndDot(float* q, float* k, float qScale, bool useL2Norm, size_t dk) {
    __m256 qSum = _mm256_setzero_ps();
    __m256 kSum = _mm256_setzero_ps();
    __m256 qkSum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dk; i += 8) {
        const __m256 qVec = _mm256_loadu_ps(q + i);
        const __m256 kVec = _mm256_loadu_ps(k + i);
        if (useL2Norm) {
            qSum = _mm256_add_ps(qSum, _mm256_mul_ps(qVec, qVec));
            kSum = _mm256_add_ps(kSum, _mm256_mul_ps(kVec, kVec));
        }
        qkSum = _mm256_add_ps(qkSum, _mm256_mul_ps(qVec, kVec));
    }
    alignas(32) float qSumValues[8];
    alignas(32) float kSumValues[8];
    alignas(32) float qkSumValues[8];
    _mm256_store_ps(qSumValues, qSum);
    _mm256_store_ps(kSumValues, kSum);
    _mm256_store_ps(qkSumValues, qkSum);
    float qSumScalar = 0.0f;
    float kSumScalar = 0.0f;
    float qkScalar = 0.0f;
    for (int lane = 0; lane < 8; ++lane) {
        qSumScalar += qSumValues[lane];
        kSumScalar += kSumValues[lane];
        qkScalar += qkSumValues[lane];
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
    const __m256 qScaleVec = _mm256_set1_ps(qNormScale);
    const __m256 kScaleVec = _mm256_set1_ps(kNormScale);
    for (i = 0; i + 8 <= dk; i += 8) {
        _mm256_storeu_ps(q + i, _mm256_mul_ps(_mm256_loadu_ps(q + i), qScaleVec));
        if (useL2Norm) {
            _mm256_storeu_ps(k + i, _mm256_mul_ps(_mm256_loadu_ps(k + i), kScaleVec));
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

} // namespace

extern "C" void _AVX_LinearAttentionInit(void* functions) {
    auto coreFunction = static_cast<MNN::CoreFunctions*>(functions);
    coreFunction->MNNDualMatVec = _AVX_MNNDualMatVec;
    coreFunction->MNNDecayRankOneUpdate = _AVX_MNNDecayRankOneUpdate;
    coreFunction->MNNFusedGatedDelta = _AVX_MNNFusedGatedDelta;
    coreFunction->MNNNormalizeQKAndDot = _AVX_MNNNormalizeQKAndDot;
}

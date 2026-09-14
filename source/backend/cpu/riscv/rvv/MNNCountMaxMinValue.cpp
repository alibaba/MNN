#include <riscv_vector.h>
#include <stddef.h>

static inline void MNNCountMaxMinValueScalar(const float* source, float* minVal, float* maxVal, size_t size) {
    if (size == 0) {
        *minVal = 0.0f;
        *maxVal = 0.0f;
        return;
    }

    float minValue = source[0];
    float maxValue = source[0];
    for (size_t i = 1; i < size; ++i) {
        const float value = source[i];
        if (value < minValue) {
            minValue = value;
        }
        if (value > maxValue) {
            maxValue = value;
        }
    }
    *minVal = minValue;
    *maxVal = maxValue;
}

void MNNCountMaxMinValue_RVV(const float* source, float* minVal, float* maxVal, size_t size) {
    constexpr size_t kScalarThreshold = 16;
    if (size <= kScalarThreshold) {
        MNNCountMaxMinValueScalar(source, minVal, maxVal, size);
        return;
    }

    // Match the scalar comparisons: a leading NaN is retained, and later
    // NaNs are ignored. Seeding from the input also preserves all-infinite data.
    float localMin = source[0];
    float localMax = source[0];
    if (localMin != localMin) {
        *minVal = localMin;
        *maxVal = localMax;
        return;
    }
    size_t offset = 1;
    while (offset < size) {
        const size_t vl = __riscv_vsetvl_e32m8(size - offset);
        const vfloat32m8_t value = __riscv_vle32_v_f32m8(source + offset, vl);
        const vfloat32m1_t minReduced = __riscv_vfredmin_vs_f32m8_f32m1(value, __riscv_vfmv_s_f_f32m1(localMin, 1), vl);
        const vfloat32m1_t maxReduced = __riscv_vfredmax_vs_f32m8_f32m1(value, __riscv_vfmv_s_f_f32m1(localMax, 1), vl);
        localMin = __riscv_vfmv_f_s_f32m1_f32(minReduced);
        localMax = __riscv_vfmv_f_s_f32m1_f32(maxReduced);
        offset += vl;
    }

    // RVV min/max order signed zeros, while scalar comparisons retain the
    // first zero encountered. Restore that sign when an extremum is zero.
    if (localMin == 0.0f || localMax == 0.0f) {
        for (size_t i = 0; i < size; ++i) {
            if (source[i] == 0.0f) {
                if (localMin == 0.0f) {
                    localMin = source[i];
                }
                if (localMax == 0.0f) {
                    localMax = source[i];
                }
                break;
            }
        }
    }

    *minVal = localMin;
    *maxVal = localMax;
}

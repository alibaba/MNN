#include <stddef.h>

void MNNPackedMatMulRemainFP32_RVV(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter,
                                   const float* postParameters, const float* bias, const float* k, const float* b);
void MNNPackedMatMulFP32_RVV(float* C, const float* A, const float* B, const size_t* parameter,
                             const float* postParameters, const float* bias, const float* k, const float* b) {
    MNNPackedMatMulRemainFP32_RVV(C, A, B, 16, parameter, postParameters, bias, k, b);
}

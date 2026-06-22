#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <cstdlib>
#include "backend/cpu/compute/CommonOptFunction.h"

void ref_add_rmsnorm(float* dst, const float* src0, const float* src1,
                     const float* gamma, float epsilon, size_t size) {
    std::vector<float> tmp(size);
    for (size_t i = 0; i < size; i++) tmp[i] = src0[i] + src1[i];
    MNNNorm(dst, tmp.data(), gamma, nullptr, epsilon, size, true);
}

int main() {
    const size_t SIZE = 2560;
    const float EPS = 1e-5f;
    const int REPEAT = 1000;
    std::vector<float> src0(SIZE), src1(SIZE), gamma(SIZE), ref_out(SIZE), fused_out(SIZE);
    srand(42);
    for (size_t i = 0; i < SIZE; i++) {
        src0[i] = (float)rand()/RAND_MAX*2-1;
        src1[i] = (float)rand()/RAND_MAX*2-1;
        gamma[i] = (float)rand()/RAND_MAX*0.5f+0.75f;
    }

    // Accuracy
    ref_add_rmsnorm(ref_out.data(), src0.data(), src1.data(), gamma.data(), EPS, SIZE);
    MNNAddAndRMSNorm(fused_out.data(), src0.data(), src1.data(), gamma.data(), EPS, SIZE);
    float max_err = 0, cos_sim_num = 0, den_a = 0, den_b = 0;
    for (size_t i = 0; i < SIZE; i++) {
        float e = fabs(ref_out[i]-fused_out[i]);
        if (e>max_err) max_err=e;
        cos_sim_num += ref_out[i]*fused_out[i];
        den_a += ref_out[i]*ref_out[i];
        den_b += fused_out[i]*fused_out[i];
    }
    printf("MaxErr: %e  CosSim: %.8f  %s\n", max_err, cos_sim_num/(sqrt(den_a)*sqrt(den_b)),
           max_err<1e-5?"PASS":"FAIL");

    // Performance
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r=0; r<REPEAT; r++) ref_add_rmsnorm(ref_out.data(),src0.data(),src1.data(),gamma.data(),EPS,SIZE);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int r=0; r<REPEAT; r++) MNNAddAndRMSNorm(fused_out.data(),src0.data(),src1.data(),gamma.data(),EPS,SIZE);
    auto t2 = std::chrono::high_resolution_clock::now();
    double ref_ms = std::chrono::duration<double,std::milli>(t1-t0).count()/REPEAT;
    double fused_ms = std::chrono::duration<double,std::milli>(t2-t1).count()/REPEAT;
    printf("Ref: %.4fms  Fused: %.4fms  Speedup: %.2fx\n", ref_ms, fused_ms, ref_ms/fused_ms);
    return 0;
}

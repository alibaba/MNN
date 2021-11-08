//
//  PluginMatMulCommon.hpp
//  MNNTests
//
//  Created by MNN on 2020/04/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_TEST_PLUGIN_PLUGIN_MATMUL_COMMON_HPP_
#define MNN_TEST_PLUGIN_PLUGIN_MATMUL_COMMON_HPP_

#include <cstring>

namespace MNN {
namespace plugin {

inline void doGemm(const int M, const int N, const int K,          // NOLINT
                   const bool transpose_x, const bool transpose_y, // NOLINT
                   const float* x, const float* y, float* out) {
    memset(out, 0, M * N * sizeof(float));
    int x_m_stride = K, x_k_stride = 1, y_k_stride = N, y_n_stride = 1;
    if (transpose_x) {
        x_m_stride = 1;
        x_k_stride = M;
    }
    if (transpose_y) {
        y_k_stride = 1;
        y_n_stride = K;
    }

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                out[m * N + n] += x[m * x_m_stride + k * x_k_stride] * // NOLINT
                                  y[k * y_k_stride + n * y_n_stride];
            }
        }
    }
}

} // namespace plugin
} // namespace MNN

#endif // MNN_TEST_PLUGIN_PLUGIN_MATMUL_COMMON_HPP_

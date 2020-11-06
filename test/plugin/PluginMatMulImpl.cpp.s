//
//  PluginMatMulImpl.cpp.s
//  MNNTests
//
//  Created by MNN on 2020/04/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "./PluginMatMulCommon.hpp"
#include "MNN/plugin/PluginKernel.hpp"
#include "MNN/plugin/PluginShapeInference.hpp"

MNN_PUBLIC int _intPluginMatMul = 10; // Just for linking successfully.

namespace MNN {
namespace plugin {

namespace shape_inference {
class PluginMatMul : public InferShapeKernel {
public:
    bool compute(InferShapeContext* ctx) override;
};

bool PluginMatMul::compute(InferShapeContext* ctx) {
    MNN_CHECK(ctx->inputs().size() == 2, // NOLINT
              "PluginMatMul needs two inputs (x and y).");
    MNN_CHECK(ctx->outputs().size() == 1, "PluginMatMul needs one output.");
    bool transpose_x = false;
    bool transpose_y = false;
    if (ctx->hasAttr("transpose_x")) {
        transpose_x = ctx->getAttr("transpose_x")->b();
    }
    if (ctx->hasAttr("transpose_y")) {
        transpose_y = ctx->getAttr("transpose_y")->b();
    }

    const auto& x = ctx->input(0)->buffer();
    const auto& y = ctx->input(1)->buffer();
    auto& output  = ctx->output(0)->buffer();

    MNN_CHECK(x.dimensions == 2, "PluginMatMul only support 2-D input.");
    MNN_CHECK(y.dimensions == 2, "PluginMatMul only support 2-D input.");

    int M = x.dim[0].extent;
    int K = x.dim[1].extent;
    int N = y.dim[1].extent;
    if (transpose_x) {
        M = x.dim[1].extent;
        K = x.dim[0].extent;
    }
    if (transpose_y) {
        N = y.dim[0].extent;
        MNN_CHECK(K == y.dim[1].extent, "K dim does not match.");
    } else {
        MNN_CHECK(K == y.dim[0].extent, "K dim does not match.");
    }
    output.dimensions    = 2;
    output.type          = x.type;
    output.dim[0].extent = M;
    output.dim[1].extent = N;
    return true /*success*/;
}
} // namespace shape_inference

namespace backend {
class PluginMatMul : public CPUComputeKernel {
public:
    bool init(CPUKernelContext*) override { return true; }
    bool compute(CPUKernelContext* ctx) override;
};

bool PluginMatMul::compute(CPUKernelContext* ctx) {
    MNN_CHECK(ctx->inputs().size() == 2, // NOLINT
              "PluginMatMul needs two inputs (x and y).");
    MNN_CHECK(ctx->outputs().size() == 1, "PluginMatMul needs one output.");
    bool transpose_x = false;
    bool transpose_y = false;
    if (ctx->hasAttr("transpose_x")) {
        transpose_x = ctx->getAttr("transpose_x")->b();
    }
    if (ctx->hasAttr("transpose_y")) {
        transpose_y = ctx->getAttr("transpose_y")->b();
    }

    const auto& x = ctx->input(0)->buffer();
    const auto& y = ctx->input(1)->buffer();
    auto& output  = ctx->output(0)->buffer();

    int M = x.dim[0].extent;
    int K = x.dim[1].extent;
    int N = y.dim[1].extent;
    if (transpose_x) {
        M = x.dim[1].extent;
        K = x.dim[0].extent;
    }
    if (transpose_y) {
        N = y.dim[0].extent;
        MNN_CHECK(K == y.dim[1].extent, "K dim does not match.");
    } else {
        MNN_CHECK(K == y.dim[0].extent, "K dim does not match.");
    }

    const float* x_data = reinterpret_cast<const float*>(x.host);
    const float* y_data = reinterpret_cast<const float*>(y.host);
    float* output_data  = reinterpret_cast<float*>(output.host);

    // Do matrix multiply.
    doGemm(M, N, K, transpose_x, transpose_y, x_data, y_data, output_data);

    return true;
}
} // namespace backend

REGISTER_PLUGIN_OP(PluginMatMul, shape_inference::PluginMatMul);
REGISTER_PLUGIN_COMPUTE_KERNEL(PluginMatMul, backend::PluginMatMul);

} // namespace plugin
} // namespace MNN

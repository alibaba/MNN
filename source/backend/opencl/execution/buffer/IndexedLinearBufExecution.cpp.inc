//
//  IndexedLinearBufExecution.cpp.inc
//  MNN
//
//  Created by MNN on 2026/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if defined(MNN_SUPPORT_TRANSFORMER_FUSE) && !defined(MNN_OPENCL_BUFFER_CLOSED)
#include "IndexedLinearBufExecution.hpp"
#include "core/IndexedLinearCommon.hpp"
#include "half.hpp"
#include <climits>

namespace MNN {
namespace OpenCL {

IndexedLinearBufExecution::IndexedLinearBufExecution(const Op* op, Backend* backend)
    : CommonExecution(backend, op),
      mInputMode(op->main_as_IndexedLinearParam()->inputMode()),
      mFuseSwiGLU(op->main_as_IndexedLinearParam()->fusedActivation() == UnaryOpOperation_SILU) {
    mValid = initResource(op);
}

IndexedLinearBufExecution::IndexedLinearBufExecution(const Op* op, Backend* backend, std::shared_ptr<Resource> resource)
    : CommonExecution(backend, op),
      mResource(std::move(resource)),
      mInputMode(op->main_as_IndexedLinearParam()->inputMode()),
      mFuseSwiGLU(op->main_as_IndexedLinearParam()->fusedActivation() == UnaryOpOperation_SILU) {}

bool IndexedLinearBufExecution::initResource(const Op* op) {
    mResource.reset(new Resource);
    auto& r = *mResource;
    if (!IndexedLinearCommon::dimensions(op, r.experts, r.inputChannels, r.outputChannels)) {
        return false;
    }
    auto backend = static_cast<OpenCLBackend*>(this->backend());
    auto runtime = backend->getOpenCLRuntime();
    const int64_t elements = (int64_t)r.inputChannels * r.outputChannels;
    if (elements > INT_MAX / 4) {
        return false;
    }
    auto allocate = [&](std::shared_ptr<cl::Buffer>& buffer, size_t bytes) {
        cl_int error = CL_SUCCESS;
        buffer.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_ONLY, std::max<size_t>(bytes, 4), nullptr, &error));
        return error == CL_SUCCESS;
    };
    for (int e = 0; e < r.experts; ++e) {
        auto data = IndexedLinearCommon::loadLinear(op, e, this->backend());
        std::vector<float> bias;
        if (!data || !IndexedLinearCommon::loadBias(op, e, bias)) {
            return false;
        }
        const bool floating = data->weightFloat.get() != nullptr;
        const int bits = floating ? (data->quan && data->quan->type() == 3 ? 16 : 32) :
                         (data->canUseInt4 ? 4 : 8);
        const int alphaComponents = !floating && data->asymmetric ? 2 : 1;
        const float* sourceAlpha = floating ? nullptr : data->getAlphaFloat();
        const int groups = floating ? 1 : data->alphaSize / (r.outputChannels * alphaComponents);
        if ((!floating && (!sourceAlpha || data->canUseInt2 || data->canUseInt3)) || groups <= 0 ||
            r.inputChannels % groups || (!floating && data->alphaSize != r.outputChannels * groups * alphaComponents)) {
            return false;
        }
        const int bytes = (elements * bits + 7) / 8;
        if (e == 0) {
            r.weightBits = bits;
            r.groups = groups;
            r.groupSize = r.inputChannels / groups;
            r.weightStride = ROUND_UP(bytes, 4);
            if (!allocate(r.weights, (size_t)r.experts * r.weightStride) ||
                !allocate(r.alpha, (size_t)r.experts * r.outputChannels * r.groups * 2 * sizeof(float)) ||
                !allocate(r.bias, (size_t)r.experts * r.outputChannels * sizeof(float))) {
                return false;
            }
        } else if (bits != r.weightBits || groups != r.groups) {
            return false;
        }
        std::vector<float> alpha(r.outputChannels * groups * 2, 0.0f);
        for (int index = 0; index < r.outputChannels * groups; ++index) {
            alpha[index * 2] = floating || !data->asymmetric ? 0.0f : sourceAlpha[index * 2];
            alpha[index * 2 + 1] = floating ? 1.0f : sourceAlpha[index * alphaComponents + alphaComponents - 1];
        }
        std::vector<half_float::half> halfWeight;
        const void* weight = floating ? (const void*)data->weightFloat.get() : (const void*)data->weight.get();
        if (bits == 16) {
            halfWeight.resize(elements);
            for (int i = 0; i < elements; ++i) halfWeight[i] = half_float::half(data->weightFloat.get()[i]);
            weight = halfWeight.data();
        }
        if ((!floating && data->weight.size() < bytes) || !weight) return false;
        auto& queue = runtime->commandQueue();
        cl_int error = queue.enqueueWriteBuffer(*r.weights, CL_TRUE, (size_t)e * r.weightStride, bytes, weight);
        error |= queue.enqueueWriteBuffer(*r.alpha, CL_TRUE, (size_t)e * alpha.size() * sizeof(float),
                                          alpha.size() * sizeof(float), alpha.data());
        error |= queue.enqueueWriteBuffer(*r.bias, CL_TRUE, (size_t)e * bias.size() * sizeof(float),
                                          bias.size() * sizeof(float), bias.data());
        if (error != CL_SUCCESS) return false;
    }
    return true;
}

bool IndexedLinearBufExecution::onClone(Backend* backend, const Op* op, Execution** dst) {
    if (!dst) return mValid;
    int experts, channels, outputs;
    if (!IndexedLinearCommon::dimensions(op, experts, channels, outputs) || experts != mResource->experts ||
        channels != mResource->inputChannels || outputs != mResource->outputChannels ||
        op->main_as_IndexedLinearParam()->inputMode() != mInputMode ||
        (op->main_as_IndexedLinearParam()->fusedActivation() == UnaryOpOperation_SILU) != mFuseSwiGLU)
        return false;
    if (!backend || backend->type() != MNN_FORWARD_OPENCL ||
        static_cast<OpenCLBackend*>(backend)->getOpenCLRuntime() !=
        static_cast<OpenCLBackend*>(this->backend())->getOpenCLRuntime()) {
        return false;
    }
    *dst = new IndexedLinearBufExecution(op, backend, mResource);
    return true;
}

ErrorCode IndexedLinearBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (inputs.size() != 2 || outputs.size() != 1 || inputs[1]->dimensions() != 2 ||
        inputs[0]->getType() != halide_type_of<float>() || outputs[0]->getType() != halide_type_of<float>() ||
        inputs[1]->getType() != halide_type_of<int32_t>()) return NOT_SUPPORT;
    const auto& r = *mResource;
    const int tokens = inputs[1]->length(0), topk = inputs[1]->length(1);
    if (tokens <= 0 || topk <= 0 || (int64_t)tokens * topk > INT_MAX) return NOT_SUPPORT;
    for (auto tensor : {inputs[0], inputs[1], outputs[0]}) {
        if (TensorUtils::getDescribe(tensor)->dimensionFormat != MNN_DATA_FORMAT_NCHW) return NOT_SUPPORT;
        int64_t stride = 1;
        for (int dimension = tensor->dimensions() - 1; dimension >= 0; --dimension) {
            if (tensor->length(dimension) < 0 || tensor->stride(dimension) != stride) return INPUT_DATA_ERROR;
            stride *= tensor->length(dimension);
            if (stride > INT_MAX) return INPUT_DATA_ERROR;
        }
    }
    const int rows = tokens * topk;
    if (inputs[0]->dimensions() != (mInputMode == 0 ? 2 : 3) || inputs[0]->length(0) != tokens ||
        (mInputMode == 1 && inputs[0]->length(1) != topk) ||
        inputs[0]->length(inputs[0]->dimensions() - 1) != r.inputChannels || outputs[0]->dimensions() != 3 ||
        outputs[0]->length(0) != tokens || outputs[0]->length(1) != topk ||
        outputs[0]->length(2) != r.outputChannels / (mFuseSwiGLU ? 2 : 1))
        return INPUT_DATA_ERROR;
    auto backend = static_cast<OpenCLBackend*>(this->backend());
    auto runtime = backend->getOpenCLRuntime();
    const auto maxItems = runtime->getMaxWorkItemSizes();
    if (maxItems.size() < 2 || maxItems[0] < 64 || maxItems[1] < 8) return NOT_SUPPORT;
    // Both choices keep K=16. A larger routed-row tile reuses each decoded
    // weight across more tokens; sparse workloads use the smaller tile.
    int tileM = tokens > 4 && (int64_t)rows >= (int64_t)r.experts * 8 &&
                runtime->MaxWorkGroupSize() >= 128 && runtime->getMaxLocalMem() >= 6208 ? 32 : 8;
    int tileN = tileM == 32 ? 64 : 32;
    int localX = tileM == 32 ? 16 : 8;
    auto buildOptions = [&]() {
        return std::set<std::string>{"-DWEIGHT_BITS=" + std::to_string(r.weightBits),
                                     "-DBANK_I=" + std::to_string(r.inputChannels),
                                     "-DBANK_O=" + std::to_string(r.outputChannels),
                                     "-DBANK_GROUPS=" + std::to_string(r.groups),
                                     "-DBANK_GROUP_SIZE=" + std::to_string(r.groupSize),
                                     "-DBANK_STRIDE=" + std::to_string(r.weightStride),
                                     "-DBANK_TOPK=" + std::to_string(topk),
                                     "-DBANK_INPUT_MODE=" + std::to_string(mInputMode),
                                     "-DFUSE_SILU_MUL=" + std::to_string(mFuseSwiGLU ? 1 : 0),
                                     "-DTILE_M=" + std::to_string(tileM),
                                     "-DTILE_N=" + std::to_string(tileN),
                                     "-DTILE_K=16"};
    };
    auto options = buildOptions();
    mUnits.clear();
    auto kernel = [&](const char* name) -> std::shared_ptr<KernelWrap> {
        return runtime->buildKernel("indexed_linear_buf", name, options, backend->getPrecision());
    };
    auto record = [&](std::shared_ptr<KernelWrap> function, std::vector<uint32_t> global, std::vector<uint32_t> local) {
        backend->recordKernel3d(function, global, local);
        Unit unit;
        unit.kernel = function;
        unit.globalWorkSize = cl::NDRange(global[0], global[1], global[2]);
        unit.localWorkSize = cl::NDRange(local[0], local[1], local[2]);
        mUnits.emplace_back(std::move(unit));
    };
    // Small-M execution avoids sorting and shares each activation across four
    // output channels while one dispatch covers all selected experts.
    if (tokens <= 4) {
        auto function = kernel("indexed_linear_gemv");
        if (!function || runtime->getMaxWorkGroupSize(function) < 64 ||
            runtime->getMaxLocalMem() < (mFuseSwiGLU ? 2048 : 1024))
            return NOT_SUPPORT;
        uint32_t arg = 0;
        cl_int error = function->get().setArg(arg++, openCLBuffer(inputs[0]));
        error |= function->get().setArg(arg++, openCLBuffer(inputs[1]));
        error |= function->get().setArg(arg++, *r.weights);
        error |= function->get().setArg(arg++, *r.alpha);
        error |= function->get().setArg(arg++, *r.bias);
        error |= function->get().setArg(arg++, openCLBuffer(outputs[0]));
        for (int value : {r.inputChannels, r.outputChannels, r.experts, topk, mInputMode, r.groups, r.groupSize, r.weightStride}) {
            error |= function->get().setArg(arg++, value);
        }
        if (error != CL_SUCCESS) return NOT_SUPPORT;
        record(function, {(uint32_t)UP_DIV(r.outputChannels / (mFuseSwiGLU ? 2 : 1), 4) * 64, (uint32_t)rows, 1},
               {64, 1, 1});
        return NO_ERROR;
    }
    auto gemm = kernel("indexed_linear_gemm");
    if (tileM == 32 && (!gemm || runtime->getMaxWorkGroupSize(gemm) < 128)) {
        tileM = 8;
        tileN = 32;
        localX = 8;
        options = buildOptions();
        gemm = kernel("indexed_linear_gemm");
    }
    if (!gemm || runtime->getMaxWorkGroupSize(gemm) < 64 || runtime->getMaxLocalMem() < 2624)
        return NOT_SUPPORT;
    const int maxTiles = UP_DIV(rows, tileM) + r.experts;
    if (rows > mRowsCapacity) {
        cl_int error = CL_SUCCESS;
        auto allocate = [&](std::shared_ptr<cl::Buffer>& buffer, size_t bytes) {
            cl_int result = CL_SUCCESS;
            buffer.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE, bytes, nullptr, &result));
            error |= result;
        };
        allocate(mCounts, r.experts * sizeof(int));
        allocate(mOffsets, (r.experts + 1) * sizeof(int));
        allocate(mSortedRows, (size_t)rows * sizeof(int));
        // Reserve for the smallest tile even when this resize selects M=32.
        // A later smaller input can select M=8 and need more tiles than before.
        allocate(mTiles, ((size_t)UP_DIV(rows, 8) + r.experts) * 2 * sizeof(int));
        if (error != CL_SUCCESS) return OUT_OF_MEMORY;
        mRowsCapacity = rows;
    }
    auto clear = kernel("indexed_route_clear");
    auto count = kernel("indexed_route_count");
    auto prefix = kernel("indexed_route_prefix");
    auto scatter = kernel("indexed_route_scatter");
    if (!clear || !count || !prefix || !scatter) return NOT_SUPPORT;
    cl_int error = clear->get().setArg(0, *mCounts);
    error |= clear->get().setArg(1, r.experts);
    error |= count->get().setArg(0, openCLBuffer(inputs[1]));
    error |= count->get().setArg(1, *mCounts);
    error |= count->get().setArg(2, rows);
    error |= count->get().setArg(3, r.experts);
    error |= prefix->get().setArg(0, *mCounts);
    error |= prefix->get().setArg(1, *mOffsets);
    error |= prefix->get().setArg(2, *mTiles);
    error |= prefix->get().setArg(3, r.experts);
    error |= prefix->get().setArg(4, maxTiles);
    error |= scatter->get().setArg(0, openCLBuffer(inputs[1]));
    error |= scatter->get().setArg(1, *mCounts);
    error |= scatter->get().setArg(2, *mOffsets);
    error |= scatter->get().setArg(3, *mSortedRows);
    error |= scatter->get().setArg(4, rows);
    error |= scatter->get().setArg(5, r.experts);
    uint32_t arg = 0;
    error |= gemm->get().setArg(arg++, openCLBuffer(inputs[0]));
    error |= gemm->get().setArg(arg++, openCLBuffer(inputs[1]));
    error |= gemm->get().setArg(arg++, *r.weights);
    error |= gemm->get().setArg(arg++, *r.alpha);
    error |= gemm->get().setArg(arg++, *r.bias);
    error |= gemm->get().setArg(arg++, openCLBuffer(outputs[0]));
    error |= gemm->get().setArg(arg++, *mOffsets);
    error |= gemm->get().setArg(arg++, *mSortedRows);
    error |= gemm->get().setArg(arg++, *mTiles);
    for (int value : {r.inputChannels, r.outputChannels, topk, mInputMode, r.groups, r.groupSize, r.weightStride}) {
        error |= gemm->get().setArg(arg++, value);
    }
    if (error != CL_SUCCESS) return NOT_SUPPORT;
    record(clear, {(uint32_t)ROUND_UP(r.experts, 64), 1, 1}, {64, 1, 1});
    record(count, {(uint32_t)ROUND_UP(rows, 64), 1, 1}, {64, 1, 1});
    record(prefix, {1, 1, 1}, {1, 1, 1});
    record(scatter, {(uint32_t)ROUND_UP(rows, 64), 1, 1}, {64, 1, 1});
    record(gemm, {(uint32_t)UP_DIV(r.outputChannels, tileN) * (uint32_t)localX, (uint32_t)maxTiles * 8, 1},
           {(uint32_t)localX, 8, 1});
    return NO_ERROR;
}

class IndexedLinearBufCreator : public OpenCLBackend::Creator {
public:
    Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                        const Op* op, Backend* backend) const override {
        // StaticModule preloads resources before shape/type inference. Validate
        // dynamic inputs in onEncode, otherwise the untyped ids force CPU fallback.
        for (auto tensor : inputs) TensorUtils::setTensorSupportPack(tensor, false);
        for (auto tensor : outputs) TensorUtils::setTensorSupportPack(tensor, false);
        OPENCL_CREATOR_CHECK(new IndexedLinearBufExecution(op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR_TRANSFORMER(IndexedLinearBufCreator, OpType_IndexedLinear, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif

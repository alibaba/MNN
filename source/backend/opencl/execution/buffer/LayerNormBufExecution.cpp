//
//  LayerNormBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/07/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/buffer/LayerNormBufExecution.hpp"

namespace MNN {
namespace OpenCL {

// Measured winners for the fused residual add; handing them to the tuner keeps Wide from sweeping the
// full grid. Only Adreno was measured, and only below the gws ceiling where the shortlist still holds.
static LwsShortlist2D binaryAddC4LwsShortlist(GpuType gpuType, uint32_t gws0) {
    if (gpuType != ADRENO || gws0 > 262144) {
        return LwsShortlist2D();
    }
    static const uint32_t smallPool[][2] = {{4, 1}, {8, 1}, {16, 1}, {32, 1}, {64, 1}, {128, 1}, {256, 1}, {32, 2}};
    static const uint32_t mediumPool[][2] = {{32, 1}, {64, 1}, {128, 1}, {256, 1}, {32, 2}};
    const uint32_t (*pool)[2] = smallPool;
    size_t count = sizeof(smallPool) / sizeof(smallPool[0]);
    if (gws0 > 1024) {
        pool = mediumPool;
        count = sizeof(mediumPool) / sizeof(mediumPool[0]);
    }
    LwsShortlist2D shortlist;
    shortlist.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        shortlist.push_back({{pool[i][0], pool[i][1]}});
    }
    return shortlist;
}

LayerNormBufExecution::LayerNormBufExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const auto* layer_norm_param = op->main_as_LayerNorm();
    mResource.reset(new LayernormResource);
    if (nullptr != layer_norm_param->axis()) {
        mResource->axis_size = layer_norm_param->axis()->size();
    }
    mResource->epsilon_ = layer_norm_param->epsilon();
    mResource->group_ = layer_norm_param->group();
    mResource->RMSNorm = layer_norm_param->useRMSNorm();
    auto bufferUnitSize =
        mOpenCLBackend->getPrecision() != BackendConfig::Precision_High ? sizeof(half_float::half) : sizeof(float);
    auto kernel =
        runtime->buildKernel("layernorm_buf", "layernorm_buf", {"-DLOCAL_SIZE=512"}, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL_CTOR(kernel);
    mResource->mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(kernel));

    mResource->has_gamma_beta_ = (layer_norm_param->gamma() && layer_norm_param->beta());
    int gammasize = 0;
    if (mResource->has_gamma_beta_) {
        MNN_ASSERT(layer_norm_param->gamma()->size() == layer_norm_param->beta()->size());
        gammasize = layer_norm_param->gamma()->size();
    }
    mResource->has_gamma_beta_ =
        mResource->has_gamma_beta_ || (layer_norm_param->external() && layer_norm_param->external()->size() > 1 &&
                                       layer_norm_param->external()->data()[1] > 0);
    if (mResource->has_gamma_beta_ && gammasize == 0) {
        gammasize = layer_norm_param->external()->data()[1] / sizeof(float);
    }

    auto staticMapAlloc = mOpenCLBackend->getStaticAllocatorMMap();
    if (mResource->has_gamma_beta_) {
        {
            auto error = CL_SUCCESS;
            int size = gammasize;
            if (mOpenCLBackend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr) {
                mResource->mGammaBuffer = staticMapAlloc.get()->allocBuffer(ALIGN_UP4(size) * bufferUnitSize);
            } else {
                mResource->mGammaBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(),
                                                             CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                                             ALIGN_UP4(size) * bufferUnitSize));
            }
            OPENCL_CHECK_PTR_CTOR(mResource->mGammaBuffer);
            if (mOpenCLBackend->getRuntime()->hint().useCachedMmap <= 1) {
                auto GammaPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
                    *(mResource->mGammaBuffer.get()), true, CL_MAP_WRITE, 0, ALIGN_UP4(size) * bufferUnitSize, nullptr,
                    nullptr, &error);
                const float* gamma_data = layer_norm_param->gamma()->data();
                if (GammaPtrCL != nullptr && error == CL_SUCCESS) {
                    if (mOpenCLBackend->getPrecision() != BackendConfig::Precision_High) {
                        for (int i = 0; i < size; i++) {
                            ((half_float::half*)GammaPtrCL)[i] = (half_float::half)(gamma_data[i]);
                        }
                        for (int i = size; i < ALIGN_UP4(size); i++) {
                            ((half_float::half*)GammaPtrCL)[i] = (half_float::half)(0.0f);
                        }
                    } else {
                        ::memset(GammaPtrCL, 0, ALIGN_UP4(size) * sizeof(float));
                        ::memcpy(GammaPtrCL, gamma_data, size * sizeof(float));
                    }
                } else {
                    MNN_ERROR("Map error GammaPtrCL == nullptr \n");
                    mValid = false;
                    return;
                }
                mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mResource->mGammaBuffer.get(),
                                                                                         GammaPtrCL);
            }
        }
        {
            auto error = CL_SUCCESS;
            int size = gammasize;
            if (mOpenCLBackend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr) {
                mResource->mBetaBuffer = staticMapAlloc.get()->allocBuffer(ALIGN_UP4(size) * bufferUnitSize);
            } else {
                mResource->mBetaBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(),
                                                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                                            ALIGN_UP4(size) * bufferUnitSize));
            }
            OPENCL_CHECK_PTR_CTOR(mResource->mBetaBuffer);
            if (mOpenCLBackend->getRuntime()->hint().useCachedMmap <= 1) {
                auto BetaPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(
                    *(mResource->mBetaBuffer.get()), true, CL_MAP_WRITE, 0, ALIGN_UP4(size) * bufferUnitSize, nullptr,
                    nullptr, &error);
                const float* beta_data = layer_norm_param->beta()->data();
                if (BetaPtrCL != nullptr && error == CL_SUCCESS) {
                    if (mOpenCLBackend->getPrecision() != BackendConfig::Precision_High) {
                        for (int i = 0; i < size; i++) {
                            ((half_float::half*)BetaPtrCL)[i] = (half_float::half)(beta_data[i]);
                        }
                        for (int i = size; i < ALIGN_UP4(size); i++) {
                            ((half_float::half*)BetaPtrCL)[i] = (half_float::half)(0.0f);
                        }
                    } else {
                        ::memset(BetaPtrCL, 0, ALIGN_UP4(size) * sizeof(float));
                        ::memcpy(BetaPtrCL, beta_data, size * sizeof(float));
                    }
                } else {
                    MNN_ERROR("Map error BetaPtrCL == nullptr \n");
                    mValid = false;
                    return;
                }
                mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mResource->mBetaBuffer.get(),
                                                                                         BetaPtrCL);
            }
        }
    }
}

LayerNormBufExecution::LayerNormBufExecution(std::shared_ptr<LayernormResource> resource, const Op* op,
                                             Backend* backend)
    : CommonExecution(backend, op) {
    mResource = resource;
    mOpenCLBackend = (OpenCLBackend*)backend;
}

bool LayerNormBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new LayerNormBufExecution(mResource, op, bn);
    return true;
}

int LayerNormBufExecution::getLocalSize(int size, int maxGroupSize) {
    int local_size = 1;
    while (local_size * 2 <= maxGroupSize && local_size * 2 <= size) {
        local_size *= 2;
    }
    return local_size;
}

void LayerNormBufExecution::prebuildOpenCLPrograms(const std::vector<Tensor*>& inputs,
                                                   const std::vector<Tensor*>& outputs) {
    if (inputs.empty()) {
        return;
    }
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto input = inputs[0];
    const bool isNC4HW4 = TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
    const int rank = input->dimensions();
    int innerSize = 1;
    for (int i = rank - mResource->axis_size; i < rank; ++i) {
        innerSize *= input->length(i);
    }
    if (mResource->group_ > 1) {
        innerSize = 1;
        for (int i = 1; i < rank; ++i) {
            innerSize *= input->length(i);
        }
        innerSize /= mResource->group_;
    }
    if (isNC4HW4) {
        innerSize = input->length(1);
    }

    const auto maxLocalSize =
        std::min(std::min(runtime->getMaxWorkItemSizes()[0], mResource->mMaxWorkGroupSize), static_cast<uint32_t>(256));
    const int localSize = getLocalSize(isNC4HW4 ? UP_DIV(innerSize, 4) : innerSize / 4, maxLocalSize);
    std::set<std::string> buildOptions = {"-DLOCAL_SIZE=" + std::to_string(localSize)};
    if (mResource->RMSNorm) {
        buildOptions.emplace("-DRMSNORM");
    }
    if (mResource->has_gamma_beta_) {
        buildOptions.emplace("-DGAMMA_BETA");
    }
    if (!isNC4HW4 && innerSize % 4 != 0) {
        buildOptions.emplace("-DPACK_LEAVE");
    }
    runtime->submitPrebuild("layernorm_buf", buildOptions, mOpenCLBackend->getPrecision(), nullptr, nullptr, true,
                            true);
    if (isNC4HW4 && inputs.size() == 2 && outputs.size() == 2) {
        runtime->submitPrebuild("layernorm_buf", {}, mOpenCLBackend->getPrecision(), nullptr, nullptr, true, true);
    }
}

ErrorCode LayerNormBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    Tensor* input = inputs[0];
    Tensor* output = outputs[0];
    auto runtime = ((OpenCLBackend*)backend())->getOpenCLRuntime();
    auto MaxLocalSize =
        std::min(std::min(runtime->getMaxWorkItemSizes()[0], mResource->mMaxWorkGroupSize), (uint32_t)256);

    const auto layout = TensorUtils::getDescribe(input)->dimensionFormat;
    bool isNC4HW4 = layout == MNN_DATA_FORMAT_NC4HW4;

    std::vector<int> inputShape = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    int rank = inputs.at(0)->dimensions();
    int outter_size = 1;
    int inner_size = 1;
    for (int i = 0; i < rank - mResource->axis_size; ++i) {
        outter_size *= inputs.at(0)->length(i);
    }
    for (int i = rank - mResource->axis_size; i < rank; ++i) {
        inner_size *= inputs.at(0)->length(i);
    }

    if (mResource->group_ > 1) {
        outter_size = inputs[0]->length(0) * mResource->group_;
        inner_size = 1;
        for (int i = 1; i < rank; i++) {
            inner_size *= inputs[0]->length(i);
        }
        inner_size /= mResource->group_;
    }

    if (isNC4HW4) {
        inner_size = inputs.at(0)->length(1);
        outter_size = 1;
        for (int i = 0; i < rank; i++) {
            if (i != 1) {
                outter_size *= inputs.at(0)->length(i);
            }
        }
    }
    bool splitBinaryLN = (isNC4HW4 && inputs.size() == 2 && outputs.size() == 2);

    int local_size;
    std::string kernelName;
    if (isNC4HW4) {
        int channelUnit = UP_DIV(inner_size, 4);
        local_size = getLocalSize(channelUnit, MaxLocalSize);
        if (splitBinaryLN) {
            // The second stage kernel (LayerNorm only)
            kernelName = "layernorm_c4_buf";
        } else {
            kernelName = "layernorm_c4_buf";
        }
    } else {
        local_size = getLocalSize(inner_size / 4, MaxLocalSize);
        kernelName = "layernorm_buf";
    }

    std::set<std::string> buildOptions;
    buildOptions.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
    if (mResource->RMSNorm) {
        buildOptions.emplace("-DRMSNORM");
    }
    if (mResource->has_gamma_beta_) {
        buildOptions.emplace("-DGAMMA_BETA");
    }
    if (!isNC4HW4 && inner_size % 4 != 0) {
        buildOptions.emplace("-DPACK_LEAVE");
    }

    if (splitBinaryLN) {
        // ---------- Two-kernel SPLIT path ----------
        int total_size_float = outter_size * ROUND_UP(inner_size, 4);
        int gws_x = UP_DIV(total_size_float, 4);
        mUnits.resize(2);

        // unit[0]: binary_add_c4_buf (1D + vload4, following binary_buf.cl pattern)
        {
            auto& u0 = mUnits[0];
            std::set<std::string> addOpts;
            u0.kernel =
                runtime->buildKernel("layernorm_buf", "binary_add_c4_buf", addOpts, mOpenCLBackend->getPrecision());
            OPENCL_CHECK_KERNEL(u0.kernel);
            std::vector<uint32_t> gwsVec = {(uint32_t)gws_x, 1u};
            uint32_t maxWGS = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(u0.kernel));
            uint32_t aidx = 0;
            cl_int aret = CL_SUCCESS;
            aret |= u0.kernel->get().setArg(aidx++, gws_x);
            aret |= u0.kernel->get().setArg(aidx++, openCLBuffer(inputs[0]));
            aret |= u0.kernel->get().setArg(aidx++, openCLBuffer(inputs[1]));
            aret |= u0.kernel->get().setArg(aidx++, openCLBuffer(outputs[0]));
            aret |= u0.kernel->get().setArg(aidx++, total_size_float);
            MNN_CHECK_CL_SUCCESS(aret, "setArg binary_add_c4_buf");
            std::vector<uint32_t> lwsVec = localWS2DDefault(gwsVec, maxWGS, runtime, "binary_add_c4_buf", u0.kernel,
                                                            mOpenCLBackend->getCLTuneLevel(), "layernorm_buf",
                                                            binaryAddC4LwsShortlist(runtime->getGpuType(), gwsVec[0]))
                                               .first;
            mOpenCLBackend->recordKernel2d(u0.kernel, gwsVec, lwsVec);
            u0.globalWorkSize = {gwsVec[0], gwsVec[1]};
            u0.localWorkSize = {lwsVec[0], lwsVec[1]};
        }

        // unit[1]: layernorm_c4_buf (reads output0 residual, writes output1 norm)
        {
            auto& u1 = mUnits[1];
            u1.kernel =
                runtime->buildKernel("layernorm_buf", "layernorm_c4_buf", buildOptions, mOpenCLBackend->getPrecision());
            OPENCL_CHECK_KERNEL(u1.kernel);
            mGWS = {(uint32_t)local_size, (uint32_t)outter_size};
            mLWS = {(uint32_t)local_size, 1};
            uint32_t lidx = 0;
            cl_int lret = CL_SUCCESS;
            lret |= u1.kernel->get().setArg(lidx++, mGWS[0]);
            lret |= u1.kernel->get().setArg(lidx++, mGWS[1]);
            lret |= u1.kernel->get().setArg(lidx++, openCLBuffer(outputs[0])); // input = residual
            lret |= u1.kernel->get().setArg(lidx++, openCLBuffer(outputs[1])); // output = norm
            lret |= u1.kernel->get().setArg(lidx++, (int32_t)inner_size);
            if (mResource->has_gamma_beta_) {
                lret |= u1.kernel->get().setArg(lidx++, *mResource->mGammaBuffer.get());
                lret |= u1.kernel->get().setArg(lidx++, *mResource->mBetaBuffer.get());
            }
            lret |= u1.kernel->get().setArg(lidx++, mResource->epsilon_);
            MNN_CHECK_CL_SUCCESS(lret, "setArg layernorm_c4_buf (SPLIT)");
            mOpenCLBackend->recordKernel2d(u1.kernel, mGWS, mLWS);
            u1.globalWorkSize = {mGWS[0], mGWS[1]};
            u1.localWorkSize = {mLWS[0], mLWS[1]};
        }
        return NO_ERROR;
    }

    // ---------- Single-kernel path (no residual add, or non-NC4HW4) ----------
    mUnits.resize(1);
    auto& unit = mUnits[0];
    unit.kernel = runtime->buildKernel("layernorm_buf", kernelName, buildOptions, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL(unit.kernel);
    mGWS = {static_cast<uint32_t>(local_size), static_cast<uint32_t>(outter_size)};
    mLWS = {static_cast<uint32_t>(local_size), 1};

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGWS[0]);
    ret |= unit.kernel->get().setArg(idx++, mGWS[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(inner_size));
    if (mResource->has_gamma_beta_) {
        ret |= unit.kernel->get().setArg(idx++, *mResource->mGammaBuffer.get());
        ret |= unit.kernel->get().setArg(idx++, *mResource->mBetaBuffer.get());
    }
    ret |= unit.kernel->get().setArg(idx++, mResource->epsilon_);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LayerNormBufExecution");
    mOpenCLBackend->recordKernel2d(unit.kernel, mGWS, mLWS);
    unit.globalWorkSize = {mGWS[0], mGWS[1]};
    unit.localWorkSize = {mLWS[0], mLWS[1]};

    return NO_ERROR;
}

class LayerNormBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~LayerNormBufCreator() = default;
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        OPENCL_CREATOR_CHECK(new LayerNormBufExecution(inputs, op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR(LayerNormBufCreator, OpType_LayerNorm, BUFFER);

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_OPENCL_BUFFER_CLOSED */

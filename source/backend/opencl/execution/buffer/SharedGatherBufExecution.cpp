#ifndef MNN_OPENCL_BUFFER_CLOSED

#include <set>

#include "backend/opencl/execution/buffer/SharedGatherBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

SharedGatherBufExecution::SharedGatherBufExecution(std::shared_ptr<ConvBufResource> resource, const Op* op,
                                                   Backend* backend)
    : CommonExecution(backend, op),
      mOpenCLBackend(static_cast<OpenCLBackend*>(backend)),
      mResource(std::move(resource)) {}

bool SharedGatherBufExecution::validResource(const std::shared_ptr<ConvBufResource>& resource) {
    if (!resource.get() || !resource->mConv1x1Opt) {
        return false;
    }
    if (!resource->mDequantScaleOffsetBuffer.get()) {
        return false;
    }
    if (resource->mUseImage) {
        if (!resource->mKernelImage.get()) {
            return false;
        }
    } else if (!resource->mKernelBuffer.get()) {
        return false;
    }
    return resource->mNumQuantBit == 4 || resource->mNumQuantBit == 8;
}

bool SharedGatherBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new SharedGatherBufExecution(mResource, op, bn);
    return true;
}

ErrorCode SharedGatherBufExecution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    mUnits.resize(1);
    auto& unit = mUnits[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto indices = inputs[0];
    auto output = outputs[0];

    if (!validResource(mResource)) {
        return NOT_SUPPORT;
    }

    const int selectSize = indices->elementSize();
    const int ic = output->length(output->dimensions() - 1);
    const int oc = mResource->mOutputChannel;
    if (selectSize <= 0 || ic <= 0 || oc <= 0) {
        return NOT_SUPPORT;
    }
    const int blockSize = mResource->mBlockSize;
    if (blockSize <= 0) {
        return NOT_SUPPORT;
    }

    std::set<std::string> buildOptions;
    if (mResource->mNumQuantBit == 8) {
        buildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT8");
    } else {
        buildOptions.emplace("-DUSE_LOW_BIT_WEIGHT_INT4");
    }
    if (mResource->mBuildOptions.find("-DASYMMETRIC") != mResource->mBuildOptions.end()) {
        buildOptions.emplace("-DASYMMETRIC");
    }
    const char* kernelName = mResource->mUseImage ? "shared_gather_quant_image" : "shared_gather_quant_buffer";
    unit.kernel =
        runtime->buildKernel("shared_gather_buf", kernelName, buildOptions, mOpenCLBackend->getPrecision(), indices,
                             output);
    OPENCL_CHECK_KERNEL(unit.kernel);

    mGWS = {(uint32_t)selectSize, (uint32_t)UP_DIV(ic, 4)};
    mLWS = {16, 16};

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, (int)mGWS[0]);
    ret |= unit.kernel->get().setArg(idx++, (int)mGWS[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    if (mResource->mUseImage) {
        ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelImage.get());
    } else {
        ret |= unit.kernel->get().setArg(idx++, *mResource->mKernelBuffer.get());
    }
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(indices));
    ret |= unit.kernel->get().setArg(idx++, *mResource->mDequantScaleOffsetBuffer.get());
    ret |= unit.kernel->get().setArg(idx++, ic);
    ret |= unit.kernel->get().setArg(idx++, oc);
    ret |= unit.kernel->get().setArg(idx++, ic / blockSize);
    ret |= unit.kernel->get().setArg(idx++, mResource->mCoef);
    MNN_CHECK_CL_SUCCESS(ret, "setArg SharedGatherBufExecution");

    mOpenCLBackend->recordKernel2d(unit.kernel, mGWS, mLWS);
    unit.globalWorkSize = {mGWS[0], mGWS[1]};
    unit.localWorkSize = {mLWS[0], mLWS[1]};

    return NO_ERROR;
}

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_OPENCL_BUFFER_CLOSED */

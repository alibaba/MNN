//
//  TopKV2BufExecution.cpp
//  MNN
//
//  OpenCL buffer-path implementation of TopKV2.
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "TopKV2BufExecution.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace OpenCL {

static const int kTopKThreadNumber = 128;
static const int kTopKLocalK = 8;
static const int kTopKCandidateNumber = kTopKThreadNumber * kTopKLocalK;

TopKV2BufExecution::TopKV2BufExecution(const MNN::Op *op, Backend *backend, int k)
    : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    mK = k;
    
    mLargest = true;
    auto param = op->main_as_TopKV2();
    if (nullptr != param) {
        mLargest = param->largest();
    }
}

ErrorCode TopKV2BufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    const int rowSize = input->length(input->dimensions() - 1);
    if (rowSize <= 0) {
        mNumRows = 0;
        return NO_ERROR;
    }
    mNumRows = input->elementSize() / rowSize;
    CommonExecution::onResize(inputs, outputs);
    return NO_ERROR;
}

ErrorCode TopKV2BufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (mNumRows <= 0) {
        return NO_ERROR;
    }
    
    MNN_ASSERT(inputs.size() >= 1);
    MNN_ASSERT(outputs.size() == 2);

    auto input = inputs[0];
    auto outputValue = outputs[0];
    auto outputIndex = outputs[1];

    const int rowSize = input->length(input->dimensions() - 1);
    const int k = mK;
    
    if (k > kTopKCandidateNumber) {
        MNN_ERROR("TopKV2: k is too large, current implementation supports k <= %d\n", kTopKCandidateNumber);
        return NOT_SUPPORT;
    }

    mUnits.resize(1);
    auto &unit = mUnits[0];

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    std::set<std::string> buildOptions;
    if (mLargest) {
        buildOptions.insert("-DSORT_DESC=1");
    }
    if (input->getType().code == halide_type_int && input->getType().bits == 32) {
        buildOptions.insert("-DIS_INT=1");
    }

    unit.kernel = runtime->buildKernel("topkv2_buf", "topkv2_buf", buildOptions, mOpenCLBackend->getPrecision());
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    mGlobalWorkSize = {
        static_cast<uint32_t>(kTopKThreadNumber),
        static_cast<uint32_t>(mNumRows),
        static_cast<uint32_t>(1),
    };

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);

    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(outputValue));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(outputIndex));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));

    ret |= unit.kernel->get().setArg(idx++, rowSize);
    ret |= unit.kernel->get().setArg(idx++, k);
    ret |= unit.kernel->get().setArg(idx++, mNumRows);

    MNN_CHECK_CL_SUCCESS(ret, "setArg TopKV2BufExecution");

    mLocalWorkSize = {kTopKThreadNumber, 1, 1};

    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);

    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize  = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};

    return NO_ERROR;
}

class TopKV2BufCreator : public OpenCLBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if (inputs.size() < 2 || outputs.size() != 2) {
            return nullptr;
        }
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            return nullptr;
        }
        
        const int k = inputs[1]->host<int32_t>()[0];

        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        OPENCL_CREATOR_CHECK(new TopKV2BufExecution(op, backend, k));
    }
};

REGISTER_OPENCL_OP_CREATOR(TopKV2BufCreator, OpType_TopKV2, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */

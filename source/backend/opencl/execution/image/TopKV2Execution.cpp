//
//  TopKV2Execution.cpp
//  MNN
//
//  OpenCL image-path implementation of TopKV2.
//

#include "TopKV2Execution.hpp"
#include "core/TensorUtils.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace OpenCL {

static const int kTopKThreadNumber = 128;
static const int kTopKLocalK = 8;
static const int kTopKCandidateNumber = kTopKThreadNumber * kTopKLocalK;

TopKV2Execution::TopKV2Execution(const MNN::Op* op, Backend* backend, int k) : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    mK = k;
    mLargest = true;
    auto param = op->main_as_TopKV2();
    if (nullptr != param) {
        mLargest = param->largest();
    }
}

ErrorCode TopKV2Execution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
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

ErrorCode TopKV2Execution::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
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

    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    // Get shape info: tensorShapeFormat returns {N, H, W, C}
    std::vector<int> inputShape = tensorShapeFormat(input);
    const int width = inputShape[2];    // W
    const int channels = inputShape[3]; // C = rowSize (for 1D/2D)
    const int channelBlocks = UP_DIV(channels, 4);

    // Build kernel with appropriate options
    std::set<std::string> buildOptions;
    if (mLargest) {
        buildOptions.insert("-DSORT_DESC=1");
    }
    auto inputType = input->getType();
    if (inputType.code == halide_type_int && inputType.bits == 32) {
        buildOptions.insert("-DIS_INT=1");
    }

    mUnits.resize(1);
    Unit& unit = mUnits[0];
    unit.kernel = runtime->buildKernel("topkv2", "topkv2_channel", buildOptions, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL(unit.kernel);

    std::vector<uint32_t> gws = {
        static_cast<uint32_t>(kTopKThreadNumber),
        static_cast<uint32_t>(mNumRows),
        static_cast<uint32_t>(1),
    };
    std::vector<uint32_t> lws = {
        static_cast<uint32_t>(kTopKThreadNumber),
        static_cast<uint32_t>(1),
        static_cast<uint32_t>(1),
    };

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, gws[0]);
    ret |= unit.kernel->get().setArg(idx++, gws[1]);
    ret |= unit.kernel->get().setArg(idx++, gws[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLImage(input));
    ret |= unit.kernel->get().setArg(idx++, openCLImage(outputValue));
    ret |= unit.kernel->get().setArg(idx++, openCLImage(outputIndex));
    ret |= unit.kernel->get().setArg(idx++, rowSize);
    ret |= unit.kernel->get().setArg(idx++, k);
    ret |= unit.kernel->get().setArg(idx++, width);
    ret |= unit.kernel->get().setArg(idx++, channelBlocks);
    MNN_CHECK_CL_SUCCESS(ret, "setArg TopKV2 topkv2_channel");

    mOpenCLBackend->recordKernel3d(unit.kernel, gws, lws);

    unit.globalWorkSize = {gws[0], gws[1], gws[2]};
    unit.localWorkSize = {lws[0], lws[1], lws[2]};

    return NO_ERROR;
}

class TopKV2Creator : public OpenCLBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        if (inputs.size() < 2 || outputs.size() != 2) {
            return nullptr;
        }
        if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
            return nullptr;
        }
        // Only support sort along channel (last dim mapped to C) for dims <= 2
        if (inputs[0]->dimensions() > 2) {
            return nullptr;
        }

        const int k = inputs[1]->host<int32_t>()[0];

        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        OPENCL_CREATOR_CHECK(new TopKV2Execution(op, backend, k));
    }
};

REGISTER_OPENCL_OP_CREATOR(TopKV2Creator, OpType_TopKV2, IMAGE);

} // namespace OpenCL
} // namespace MNN

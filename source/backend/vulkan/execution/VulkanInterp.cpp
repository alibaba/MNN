//
//  VulkanInterp.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanInterp.hpp"

namespace MNN {

VulkanInterp::VulkanInterp(const Op* op, Backend* bn) : VulkanResize(bn, 1, 1, op->main_as_Interp()->resizeType()) {
    auto interpParam = op->main_as_Interp();
    mAlignCorners    = interpParam->alignCorners();
}

VulkanInterp::~VulkanInterp() {
}

ErrorCode VulkanInterp::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                 const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto& ib    = input->buffer();
    auto& ob    = output->buffer();

    int iw = ib.dim[3].extent, ow = ob.dim[3].extent;
    int ih = ib.dim[2].extent, oh = ob.dim[2].extent;

    float xScale = 1;
    float yScale = 1;
    if (mAlignCorners) {
        yScale = (float)(ih - 1) / (float)(oh - 1);
        xScale = (float)(iw - 1) / (float)(ow - 1);
    } else {
        yScale = (float)(ih) / (float)(oh);
        xScale = (float)(iw) / (float)(ow);
    }

    encodeImpl(input, output, xScale, yScale, cmdBuffer);

    return NO_ERROR;
}

class VulkanInterpCreator : public VulkanBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanInterp(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Interp, new VulkanInterpCreator);
    return true;
}();

} // namespace MNN

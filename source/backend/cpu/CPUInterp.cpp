//
//  CPUInterp.cpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUInterp.hpp"
#include "CPUBackend.hpp"
#include "CPUResize.hpp"

namespace MNN {

CPUInterp::CPUInterp(Backend *backend, float widthScale, float heightScale, int resizeType, bool AlignCorners)
    : Execution(backend),
      mWidthScale(widthScale),
      mHeightScale(heightScale),
      mResizeType(resizeType),
      mAlignCorners(AlignCorners) {
    // nothing to do
}

ErrorCode CPUInterp::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto &input  = inputs[0]->buffer();
    auto &output = outputs[0]->buffer();

    int iw = input.dim[3].extent, ow = output.dim[3].extent;
    int ih = input.dim[2].extent, oh = output.dim[2].extent;

    if (mAlignCorners) {
        mHeightScale = (float)(ih - 1) / (float)(oh - 1);
        mWidthScale  = (float)(iw - 1) / (float)(ow - 1);
    } else {
        mHeightScale = (float)(ih) / (float)(oh);
        mWidthScale  = (float)(iw) / (float)(ow);
    }

    if (mResizeType == 1) {
        // Nearstneighbor
        CPUReiseNearstneighborC4(input, output, mWidthScale, mHeightScale);
    } else if (mResizeType == 2) {
        // bilinear
        CPUResizeBilinearC4(input, output, mWidthScale, mHeightScale);
    } else if (mResizeType == 3) {
        // cubic
        CPUResizeCubicC4(input, output);
    } else {
        return NOT_SUPPORT;
        // not supported
    }
    return NO_ERROR;
}

class CPUInterpCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto interp = op->main_as_Interp();
        return new CPUInterp(backend, interp->widthScale(), interp->heightScale(), interp->resizeType(),
                             interp->alignCorners());
    }
};
REGISTER_CPU_OP_CREATOR(CPUInterpCreator, OpType_Interp);

} // namespace MNN

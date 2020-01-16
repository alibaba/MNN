//
//  CPUInterp.cpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUInterp.hpp"
#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPUResize.hpp"

namespace MNN {

static int CLAMP(int v, int min, int max) {
    if ((v) < min) {
        (v) = min;
    } else if ((v) > max) {
        (v) = max;
    }
    return v;
}

CPUInterp::CPUInterp(Backend *backend, float widthScale, float heightScale, int resizeType, bool AlignCorners, bool halfPixelCenters)
    : CPUResizeCommon(backend),
      mWidthScale(widthScale),
      mHeightScale(heightScale),
      mResizeType(resizeType),
      mAlignCorners(AlignCorners),
      mHalfPixelCenters(halfPixelCenters) {
    // nothing to do
}

CPUInterp::~CPUInterp() {
}

ErrorCode CPUInterp::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto &input  = inputs[0]->buffer();
    auto &output = outputs[0]->buffer();

    if (mResizeType == 1) {
        // Nearstneighbor
        CPUResizeNearestneighborC4(input, output, mWidthScale, mHeightScale);
    } else if (mResizeType == 2) {
        // bilinear
        CPUResizeBilinearC4(input, output, mWidthPosition.host<int>(), mWidthFactor.host<float>(),
                            mHeightPosition.host<int>(), mHeightFactor.host<float>(), mLineBuffer.host<float>(),
                            ((CPUBackend *)backend())->threadNumber());
    } else if (mResizeType == 3) {
        // cubic
        CPUResizeCubicC4(input, output);
    } else {
        return NOT_SUPPORT;
        // not supported
    }
    return NO_ERROR;
}

ErrorCode CPUInterp::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const int inW  = inputs[0]->buffer().dim[3].extent;
    const int inH  = inputs[0]->buffer().dim[2].extent;
    const int outW = outputs[0]->buffer().dim[3].extent;
    const int outH = outputs[0]->buffer().dim[2].extent;

    if (mAlignCorners) {
        mHeightScale = (float)(inH - 1) / (float)(outH - 1);
        mWidthScale  = (float)(inW - 1) / (float)(outW - 1);
    } else {
        mHeightScale = (float)(inH) / (float)(outH);
        mWidthScale  = (float)(inW) / (float)(outW);
    }

    const float xScaling = mWidthScale;
    const float yScaling = mHeightScale;

    mWidthPosition.buffer().dim[0].extent = 2 * outW;
    mWidthPosition.buffer().dimensions    = 1;
    mWidthPosition.setType(DataType_DT_INT32);
    backend()->onAcquireBuffer(&mWidthPosition, Backend::DYNAMIC_SEPERATE);

    mWidthFactor.buffer().dim[0].extent = outW;
    mWidthFactor.buffer().dimensions    = 1;
    mWidthFactor.setType(DataType_DT_FLOAT);
    backend()->onAcquireBuffer(&mWidthFactor, Backend::DYNAMIC_SEPERATE);

    auto _wPosition = mWidthPosition.host<int>();
    auto _wFactor   = mWidthFactor.host<float>();

    // Compute Line Position
    for (int x = 0; x < outW; ++x) {
        float srcX;
        if (mHalfPixelCenters) {
            srcX = (x + 0.5) * xScaling - 0.5;
        } else {
            srcX = x * xScaling;
        }
        int x1         = floor(srcX);
        float x2Factor = srcX - x1;

        _wFactor[x]           = x2Factor;
        _wPosition[2 * x + 0] = CLAMP(x1, 0, inW - 1);
        _wPosition[2 * x + 1] = CLAMP(x1 + 1, 0, inW - 1);
    }

    mHeightPosition.buffer().dim[0].extent = 2 * outH;
    mHeightPosition.buffer().dimensions    = 1;
    mHeightPosition.setType(DataType_DT_INT32);
    backend()->onAcquireBuffer(&mHeightPosition, Backend::DYNAMIC_SEPERATE);

    mHeightFactor.buffer().dim[0].extent = outH;
    mHeightFactor.buffer().dimensions    = 1;
    mHeightFactor.setType(DataType_DT_FLOAT);
    backend()->onAcquireBuffer(&mHeightFactor, Backend::DYNAMIC_SEPERATE);

    auto _hPosition = mHeightPosition.host<int>();
    auto _hFactor   = mHeightFactor.host<float>();

    for (int y = 0; y < outH; ++y) {
        float srcY;
        if (mHalfPixelCenters) {
            srcY = (y + 0.5) * yScaling - 0.5;
        } else {
            srcY = y * yScaling;
        }
        int y1         = floor(srcY);
        float y2Factor = srcY - y1;

        _hFactor[y]           = y2Factor;
        _hPosition[2 * y + 0] = CLAMP(y1, 0, inH - 1);
        _hPosition[2 * y + 1] = CLAMP(y1 + 1, 0, inH - 1);
    }

    int threadNumber = ((CPUBackend *)backend())->threadNumber();

    mLineBuffer.buffer().dim[0].extent = 2 * 4 * outW * threadNumber;
    mLineBuffer.buffer().dimensions    = 1;
    mLineBuffer.setType(DataType_DT_FLOAT);
    backend()->onAcquireBuffer(&mLineBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mLineBuffer, Backend::DYNAMIC);

    return NO_ERROR;
}

class CPUInterpCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto interp = op->main_as_Interp();
        return new CPUInterp(backend, interp->widthScale(), interp->heightScale(), interp->resizeType(),
                             interp->alignCorners(), interp->halfPixelCenters());
    }
};
REGISTER_CPU_OP_CREATOR(CPUInterpCreator, OpType_Interp);

} // namespace MNN

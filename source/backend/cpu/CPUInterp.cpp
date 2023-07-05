//
//  CPUInterp.cpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUInterp.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPUResize.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include <math.h>
#include "core/Macro.h"

namespace MNN {

CPUInterp::CPUInterp(Backend *backend, int resizeType,
                     float widthScale, float heightScale, float widthOffset, float heightOffset)
    : CPUResizeCommon(backend),
      mResizeType(resizeType),
      mWidthScale(widthScale),
      mHeightScale(heightScale),
      mWidthOffset(widthOffset),
      mHeightOffset(heightOffset) {
    // nothing to do
}

CPUInterp::~CPUInterp() {
    if (mInit && mResizeType == 2) {
        backend()->onReleaseBuffer(&mWidthPosition, Backend::STATIC);
        backend()->onReleaseBuffer(&mWidthFactor, Backend::STATIC);
        backend()->onReleaseBuffer(&mHeightPosition, Backend::STATIC);
        backend()->onReleaseBuffer(&mHeightFactor, Backend::STATIC);
    }
}

ErrorCode CPUInterp::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto channel_input = inputs[0]->channel();
    auto plane_in = inputs[0]->width() * inputs[0]->height() * inputs[0]->batch();
    auto plane_out = outputs[0]->width() * outputs[0]->height() * outputs[0]->batch();
    auto depth = UP_DIV(channel_input, core->pack);
    
    bool interpInt8 = CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1;
    if (!interpInt8) {
        switch (mResizeType) {
            case 1:
                CPUResizeNearestneighborC4<float>(inputs, outputs, mWidthScale, mHeightScale, mWidthOffset, mHeightOffset);
                break;
            case 2:
                CPUResizeBilinearC4<float, float>(CPUBilinearSampleC4, CPUBilinearLineC4, inputs, outputs, mWidthPosition.host<int>(),
                                                  mWidthFactor.host<float>(), mHeightPosition.host<int>(), mHeightFactor.host<float>(),
                                                  mLineBuffer.host<float>(), ((CPUBackend *)backend())->threadNumber(), &mInputQuantZero, &mOutputQuantZero);
                break;
            case 3:
                CPUResizeCubicC4<float>(MNNCubicSampleC4, MNNCubicLineC4, inputs, outputs, mWidthScale, mHeightScale, mWidthOffset, mHeightOffset, &mInputQuantZero, &mOutputQuantZero, mOutputQuantMIn, mOutputQuantMax);
                break;
            case 4:
                CPUResizeNearestneighborRoundC4<float>(inputs, outputs, mWidthScale, mHeightScale, mWidthOffset, mHeightOffset);
                break;
            default:
                return NOT_SUPPORT;
        }
        return NO_ERROR;
    }

    // InterpInt8.
    std::vector<Tensor *> int8ExeInputs, int8ExeOutputs;
    int8ExeInputs = {inputs[0]};
    int8ExeOutputs = {outputs[0]};

    // Pack
    if ((mResizeType == 1 || mResizeType == 2) && (core->pack == 4)) {
        MNNPackInt8C2Origin(mInputTemp.get()->host<float>(), inputs[0]->host<float>(), plane_in, depth, plane_in);
        int8ExeInputs = {mInputTemp.get()};
        int8ExeOutputs = {mOutputTemp.get()};
    } else if ((mResizeType == 3 || mResizeType == 4)) {
        if (core->pack == 4) {
            MNNPackC4Origin(mInputTemp.get()->host<float>(), inputs[0]->host<float>(), plane_in, depth, plane_in);
            int8ExeInputs = {mInputTemp.get()};
            int8ExeOutputs = {mOutputTemp.get()};
        } else if (core->pack == 8) {
            MNNPackC2Origin(mInputTemp.get()->host<double>(), inputs[0]->host<double>(), plane_in, depth, plane_in);
            int8ExeInputs = {mInputTemp.get()};
            int8ExeOutputs = {mOutputTemp.get()};
        }
    }
    // execute interpInt8
    switch (mResizeType) {
        case 1:
            CPUResizeNearestneighborC4<int8_t>(int8ExeInputs, int8ExeOutputs, mWidthScale, mHeightScale, mWidthOffset, mHeightOffset);
            break;
        case 2:
            CPUResizeBilinearC4<int8_t, int16_t>(MNNBilinearSampleC8, MNNBilinearLineC8, int8ExeInputs, int8ExeOutputs, mWidthPosition.host<int>(), mWidthFactor.host<float>(), mHeightPosition.host<int>(), mHeightFactor.host<float>(), mLineBuffer.host<int16_t>(), ((CPUBackend *)backend())->threadNumber(), &mInputQuantZero, &mOutputQuantZero);
            break;
        case 3:
            CPUResizeCubicC4<int8_t>(MNNCubicSampleC16, MNNCubicLineC16, int8ExeInputs, int8ExeOutputs, mWidthScale, mHeightScale, mWidthOffset, mHeightOffset, &mInputQuantZero, &mOutputQuantZero, mOutputQuantMIn, mOutputQuantMax);
            break;
        case 4:
            CPUResizeNearestneighborRoundC4<int8_t>(int8ExeInputs, int8ExeOutputs, mWidthScale, mHeightScale, mWidthOffset, mHeightOffset);
            break;
        default:
            return NOT_SUPPORT;
    }
    // Unpack
    if ((mResizeType == 1 || mResizeType == 2) && (core->pack == 4)) { // pack=8 -> pack=4
        MNNUnpackInt8C2Origin(outputs[0]->host<float>(), mOutputTemp.get()->host<float>(), plane_out, depth, plane_out);
    } else if ((mResizeType == 3 || mResizeType == 4)) { // pack=16 -> pack=4
        if (core->pack == 4) {
            MNNUnpackC4Origin(outputs[0]->host<float>(), mOutputTemp.get()->host<float>(), plane_out, depth, plane_out);
        } else if (core->pack == 8) {
            MNNUnpackC2Origin(outputs[0]->host<double>(), mOutputTemp.get()->host<double>(), plane_out, depth, plane_out);
        }
    }

    return NO_ERROR;
}

ErrorCode CPUInterp::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const int inW  = inputs[0]->width();
    const int inH  = inputs[0]->height();
    const int outW = outputs[0]->width();
    const int outH = outputs[0]->height();
    int packInt8 = 8;
    if (mResizeType == 3 || mResizeType == 4) {
        packInt8 = 16;
    }
    bool useInt8 = (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) && (CPUBackend::getDataType(outputs[0]) == DataType_DT_INT8 || outputs[0]->getType().bytes() == 1);
    if (useInt8) {
        mInputTemp.reset(Tensor::createDevice<int8_t>({inputs[0]->batch(), inH, inW, UP_DIV(inputs[0]->channel(), packInt8) * packInt8}));
        mOutputTemp.reset(Tensor::createDevice<int8_t>({outputs[0]->batch(), outH, outW, UP_DIV(outputs[0]->channel(), packInt8) * packInt8}));
        bool allocSucc = backend()->onAcquireBuffer(mInputTemp.get(), Backend::DYNAMIC);
        allocSucc      = allocSucc && backend()->onAcquireBuffer(mOutputTemp.get(), Backend::DYNAMIC);
        if (!allocSucc) {
            return OUT_OF_MEMORY;
        }
        mInputQuantZero = TensorUtils::getQuantInfo(inputs[0])[1];
        mOutputQuantZero = TensorUtils::getQuantInfo(outputs[0])[1];
        mOutputQuantMIn = TensorUtils::getQuantInfo(outputs[0])[2];
        mOutputQuantMax = TensorUtils::getQuantInfo(outputs[0])[3];
    }

    if (mResizeType != 2) {
        if (mInputTemp.get()) {
            backend()->onReleaseBuffer(mInputTemp.get(), Backend::DYNAMIC);
            backend()->onReleaseBuffer(mOutputTemp.get(), Backend::DYNAMIC);
        }
        return NO_ERROR;
    }
    const float xScaling = mWidthScale;
    const float yScaling = mHeightScale;

    mWidthPosition.buffer().dim[0].extent = 2 * outW;
    mWidthPosition.buffer().dimensions    = 1;
    mWidthPosition.setType(DataType_DT_INT32);

    mWidthFactor.buffer().dim[0].extent = outW;
    mWidthFactor.buffer().dimensions    = 1;
    mWidthFactor.setType(DataType_DT_FLOAT);

    mHeightPosition.buffer().dim[0].extent = 2 * outH;
    mHeightPosition.buffer().dimensions    = 1;
    mHeightPosition.setType(DataType_DT_INT32);

    mHeightFactor.buffer().dim[0].extent = outH;
    mHeightFactor.buffer().dimensions    = 1;
    mHeightFactor.setType(DataType_DT_FLOAT);
    bool res = backend()->onAcquireBuffer(&mWidthPosition, Backend::STATIC);
    res = res && backend()->onAcquireBuffer(&mWidthFactor, Backend::STATIC);
    res = res && backend()->onAcquireBuffer(&mHeightPosition, Backend::STATIC);
    res = res && backend()->onAcquireBuffer(&mHeightFactor, Backend::STATIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    auto _wPosition = mWidthPosition.host<int>();
    auto _wFactor   = mWidthFactor.host<float>();

    // Compute Line Position
    for (int x = 0; x < outW; ++x) {
        float srcX = x * xScaling + mWidthOffset;
        int x1         = floor(srcX);
        float x2Factor = srcX - x1;

        _wFactor[x]           = x2Factor;
        _wPosition[2 * x + 0] = CLAMP(x1, 0, inW - 1);
        _wPosition[2 * x + 1] = CLAMP(x1 + 1, 0, inW - 1);
    }

    auto _hPosition = mHeightPosition.host<int>();
    auto _hFactor   = mHeightFactor.host<float>();

    for (int y = 0; y < outH; ++y) {
        float srcY = y * yScaling + mHeightOffset;
        int y1         = floor(srcY);
        float y2Factor = srcY - y1;

        _hFactor[y]           = y2Factor;
        _hPosition[2 * y + 0] = CLAMP(y1, 0, inH - 1);
        _hPosition[2 * y + 1] = CLAMP(y1 + 1, 0, inH - 1);
    }

    int threadNumber = ((CPUBackend *)backend())->threadNumber();

    mLineBuffer.buffer().dim[0].extent = 2 * 4 * outW * threadNumber;
    mLineBuffer.buffer().dimensions    = 1;
    if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) {
        mLineBuffer.setType(DataType_DT_INT16);
        mLineBuffer.buffer().dim[0].extent = 2 * packInt8 * outW * threadNumber;
    } else {
        mLineBuffer.setType(DataType_DT_FLOAT);
    }
    res = backend()->onAcquireBuffer(&mLineBuffer, Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(&mLineBuffer, Backend::DYNAMIC);
    if (mInputTemp.get()) {
        backend()->onReleaseBuffer(mInputTemp.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mOutputTemp.get(), Backend::DYNAMIC);
    }
    return NO_ERROR;
}

class CPUInterpCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto interp = op->main_as_Interp();
        return new CPUInterp(backend, interp->resizeType(),
                   interp->widthScale(), interp->heightScale(), interp->widthOffset(), interp->heightOffset());
    }
};
REGISTER_CPU_OP_CREATOR(CPUInterpCreator, OpType_Interp);

} // namespace MNN

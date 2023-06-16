//
//  CPUInterp.cpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUInterp3D.hpp"
#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPUResize.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/TensorUtils.hpp"
#include "core/Macro.h"
namespace MNN {

CPUInterp3D::CPUInterp3D(Backend *backend, int resizeType,
                     float widthScale, float heightScale, float depthScale,
                     float widthOffset, float heightOffset, float depthOffset)
    : CPUResizeCommon(backend),
      mResizeType(resizeType),
      mWidthScale(widthScale),
      mHeightScale(heightScale),
      mDepthScale(depthScale),
      mWidthOffset(widthOffset),
      mHeightOffset(heightOffset),
      mDepthOffset(depthOffset) {
    // nothing to do
}

CPUInterp3D::~CPUInterp3D() {
    if (mInit && mResizeType == 2) {
        backend()->onReleaseBuffer(&mWidthPosition, Backend::STATIC);
        backend()->onReleaseBuffer(&mWidthFactor, Backend::STATIC);
        backend()->onReleaseBuffer(&mHeightPosition, Backend::STATIC);
        backend()->onReleaseBuffer(&mHeightFactor, Backend::STATIC);
        backend()->onReleaseBuffer(&mDepthPosition, Backend::STATIC);
        backend()->onReleaseBuffer(&mDepthFactor, Backend::STATIC);
    }
}
//TODO: wtd interp3d
ErrorCode CPUInterp3D::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto channel_input = inputs[0]->channel();
    int inD = inputs[0]->buffer().dim[2].extent;
    int outD = outputs[0]->buffer().dim[2].extent;
    auto plane_in = inD * inputs[0]->width() * inputs[0]->height() * inputs[0]->batch();
    auto plane_out = outD * outputs[0]->width() * outputs[0]->height() * outputs[0]->batch();
    auto depth = UP_DIV(channel_input, core->pack);
    if (mResizeType == 1) {
        // Nearstneighbor
        if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) { // int8_t
            if (core->pack == 8) {
                MNNPackC2Origin(mInputTemp.get()->host<double>(), inputs[0]->host<double>(), plane_in, depth, plane_in);
                CPUResizeNearestneighborC4<int8_t>({mInputTemp.get()}, {mOutputTemp.get()}, mWidthScale, mHeightScale, mWidthOffset, mHeightOffset);
                MNNUnpackC2Origin(outputs[0]->host<double>(), mOutputTemp.get()->host<double>(), plane_out, depth, plane_out);
            }
            else if (core->pack == 4) {
                MNNPackC4Origin(mInputTemp.get()->host<float>(), inputs[0]->host<float>(), plane_in, depth, plane_in);
                CPUResizeNearestneighborC4<int8_t>({mInputTemp.get()}, {mOutputTemp.get()}, mWidthScale, mHeightScale, mWidthOffset, mHeightOffset);
                MNNUnpackC4Origin(outputs[0]->host<float>(), mOutputTemp.get()->host<float>(), plane_out, depth, plane_out);
            }
            else if (core->pack == 16) {
                CPUResizeNearestneighborC4<int8_t>(inputs, outputs, mWidthScale, mHeightScale, mWidthOffset, mHeightOffset);
            }
        } else {
            CPUResizeNearestneighbor3DC4<float>(inputs, outputs, mWidthScale, mHeightScale, mDepthScale,
                                                 mWidthOffset, mHeightOffset, mDepthOffset);
        }
        
    } else if (mResizeType == 2) {
        // bilinear
        //CPUResizeBilinearC4(input, output, mWidthPosition.host<int>(), mWidthFactor.host<float>(),
        //                    mHeightPosition.host<int>(), mHeightFactor.host<float>(), mLineBuffer.host<float>(),
        //                    ((CPUBackend *)backend())->threadNumber());
        MNN_ERROR("Bilinear interpolation is not implemented in interp3D. Do nothing...");
    } else if (mResizeType == 3) {
        // cubic
        //CPUResizeCubicC4(input, output, mWidthScale, mHeightScale, mWidthOffset, mHeightOffset);
        MNN_ERROR("cubic interpolation is not implemented in interp3D. Do nothing...");
    } else if (mResizeType == 4) {
        // Nearstneighbor
        if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) { // int8_t
            if (core->pack == 8) {
                MNNPackC2Origin(mInputTemp.get()->host<double>(), inputs[0]->host<double>(), plane_in, depth, plane_in);
                CPUResizeNearestneighbor3DRoundC4<int8_t>({mInputTemp.get()}, {mOutputTemp.get()}, mWidthScale, mHeightScale, mDepthScale, mWidthOffset, mHeightOffset, mDepthOffset);
                MNNUnpackC2Origin(outputs[0]->host<double>(), mOutputTemp.get()->host<double>(), plane_out, depth, plane_out);
            }
            else if (core->pack == 4) {
                MNNPackC4Origin(mInputTemp.get()->host<float>(), inputs[0]->host<float>(), plane_in, depth, plane_in);
                CPUResizeNearestneighbor3DRoundC4<int8_t>({mInputTemp.get()}, {mOutputTemp.get()}, mWidthScale, mHeightScale, mDepthScale, mWidthOffset, mHeightOffset, mDepthOffset);
                MNNUnpackC4Origin(outputs[0]->host<float>(), mOutputTemp.get()->host<float>(), plane_out, depth, plane_out);
            }
            else if (core->pack == 16) {
                CPUResizeNearestneighbor3DRoundC4<int8_t>(inputs, outputs, mWidthScale, mHeightScale, mDepthScale, mWidthOffset, mHeightOffset, mDepthOffset);
            }
        } else {
            CPUResizeNearestneighbor3DRoundC4<float>(inputs, outputs, mWidthScale, mHeightScale, mDepthScale, mWidthOffset, mHeightOffset, mDepthOffset);
        }
    } else {
        return NOT_SUPPORT;
    }
    return NO_ERROR;
}

ErrorCode CPUInterp3D::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const int inW  = inputs[0]->buffer().dim[4].extent;
    const int inH  = inputs[0]->buffer().dim[3].extent;
    const int inD  = inputs[0]->buffer().dim[2].extent;
    const int outW = outputs[0]->buffer().dim[4].extent;
    const int outH = outputs[0]->buffer().dim[3].extent;
    const int outD = outputs[0]->buffer().dim[2].extent;
    const float xScaling = mWidthScale;
    const float yScaling = mHeightScale;
    const float zScaling = mDepthScale;
    
    mInputTemp.reset(Tensor::createDevice<int8_t>({inputs[0]->batch(), UP_DIV(inputs[0]->channel(), 16) * 16, inD, inH, inW}));
    mOutputTemp.reset(Tensor::createDevice<int8_t>({outputs[0]->batch(), UP_DIV(outputs[0]->channel(), 16) * 16,outD, outH, outW}));
    bool allocSucc = backend()->onAcquireBuffer(mInputTemp.get(), Backend::DYNAMIC);
    allocSucc      = allocSucc && backend()->onAcquireBuffer(mOutputTemp.get(), Backend::DYNAMIC);
    if (!allocSucc) {
        return OUT_OF_MEMORY;
    }
    if (mResizeType != 2) {
        if (mInputTemp.get()) {
            backend()->onReleaseBuffer(mInputTemp.get(), Backend::DYNAMIC);
            backend()->onReleaseBuffer(mOutputTemp.get(), Backend::DYNAMIC);
        }
        return NO_ERROR;
    }

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

    mDepthPosition.buffer().dim[0].extent = 2 * outD;
    mDepthPosition.buffer().dimensions    = 1;
    mDepthPosition.setType(DataType_DT_INT32);

    mDepthFactor.buffer().dim[0].extent = outD;
    mDepthFactor.buffer().dimensions    = 1;
    mDepthFactor.setType(DataType_DT_FLOAT);

    bool res = backend()->onAcquireBuffer(&mWidthPosition, Backend::STATIC);
    res = res && backend()->onAcquireBuffer(&mWidthFactor, Backend::STATIC);
    res = res && backend()->onAcquireBuffer(&mHeightPosition, Backend::STATIC);
    res = res && backend()->onAcquireBuffer(&mHeightFactor, Backend::STATIC);
    res = res && backend()->onAcquireBuffer(&mDepthPosition, Backend::STATIC);
    res = res && backend()->onAcquireBuffer(&mDepthFactor, Backend::STATIC);
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

    auto _dPosition = mDepthPosition.host<int>();
    auto _dFactor   = mDepthFactor.host<float>();

    for (int z = 0; z < outD; ++z) {
        float srcZ = z * zScaling + mDepthOffset;
        int z1         = floor(srcZ);
        float z2Factor = srcZ - z1;

        _dFactor[z]           = z2Factor;
        _dPosition[2 * z + 0] = CLAMP(z1, 0, inD - 1);
        _dPosition[2 * z + 1] = CLAMP(z1 + 1, 0, inD - 1);
    }

    int threadNumber = ((CPUBackend *)backend())->threadNumber();
    //TODO line buffer??
    mLineBuffer.buffer().dim[0].extent = 2 * 4 * outW * threadNumber;
    mLineBuffer.buffer().dimensions    = 1;
    mLineBuffer.setType(DataType_DT_FLOAT);
    res = backend()->onAcquireBuffer(&mLineBuffer, Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(&mLineBuffer, Backend::DYNAMIC);

    return NO_ERROR;
}

class CPUInterp3DCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const {
        auto interp3D = op->main_as_Interp();
        return new CPUInterp3D(backend, interp3D->resizeType(),
                               interp3D->widthScale(), interp3D->heightScale(), interp3D->depthScale(),
                               interp3D->widthOffset(), interp3D->heightOffset(), interp3D->depthOffset());
    }
};
REGISTER_CPU_OP_CREATOR(CPUInterp3DCreator, OpType_Interp3D);

} // namespace MNN

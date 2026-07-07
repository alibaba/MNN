//
//  QNNInterp.cpp
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "QNNInterp.hpp"
#include "QnnOpDef.h"

namespace MNN {
namespace QNN {
#ifdef ENABLE_QNN_ONLINE_FINALIZE

ErrorCode QNNInterp::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mParams.clear();
    mInputs.clear();
    mOutputs.clear();

    auto interpParam = mOp->main_as_Interp();
    int resizeType = interpParam->resizeType();
    bool alignCorners = interpParam->alignCorners();
    bool halfPixelCenters = interpParam->halfPixelCenters();

    // MNN resizeType: 1=nearest, 2=bilinear, 3=cubic, 4=nearest_round
    // Use QNN "Resize" op which supports interpolation_mode + transformation_mode

    mNodeType = "Resize";

    // Determine interpolation mode
    uint32_t interpolationMode;
    if (resizeType == 1 || resizeType == 4) {
        interpolationMode = QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST;
    } else if (resizeType == 2) {
        interpolationMode = QNN_OP_RESIZE_INTERPOLATION_MODE_LINEAR;
    } else if (resizeType == 3) {
        interpolationMode = QNN_OP_RESIZE_INTERPOLATION_MODE_CUBIC;
    } else {
        interpolationMode = QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST;
    }

    // Determine transformation mode
    uint32_t transformationMode;
    if (alignCorners) {
        transformationMode = QNN_OP_RESIZE_TRANSFORMATION_MODE_ALIGN_CORNERS;
    } else if (halfPixelCenters) {
        transformationMode = QNN_OP_RESIZE_TRANSFORMATION_MODE_HALF_PIXEL;
    } else {
        transformationMode = QNN_OP_RESIZE_TRANSFORMATION_MODE_ASYMMETRIC;
    }

    this->createParamScalar("interpolation_mode", interpolationMode);
    this->createParamScalar("transformation_mode", transformationMode);
    this->createParamScalar("exclude_outside", (uint32_t)0);

    if (resizeType == 1 || resizeType == 4) {
        uint32_t nearestMode = (resizeType == 4)
            ? QNN_OP_RESIZE_NEAREST_MODE_ROUND_PREFER_FLOOR
            : QNN_OP_RESIZE_NEAREST_MODE_FLOOR;
        this->createParamScalar("nearest_mode", nearestMode);
    }

    if (resizeType == 3) {
        float cubicCoeff = interpParam->cubicCoeffA();
        this->createParamScalar("cubic_coeff", cubicCoeff);
    }

    // Add params
    for (int i = 0; i < mParamScalarWrappers.size(); i++) {
        mParams.push_back(*(mParamScalarWrappers[i]->getNativeParam()));
    }

    mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
    mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));

    mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(), mPackageName.c_str(),
                             mNodeType.c_str(), mParams, mInputs, mOutputs);

    return NO_ERROR;
}

class QNNInterpCreator : public QnnBackend::Creator {
public:
    virtual QNNCommonExecution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                         const MNN::Op *op, Backend *backend) const override {
        return new QNNInterp(backend, op);
    }
};

REGISTER_QNN_OP_CREATOR(QNNInterpCreator, OpType_Interp)
#endif
} // end namespace QNN
} // end namespace MNN

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
    if (!mBackend->isDedicatedQnnSession()) {
        auto interpParam = mOp->main_as_Interp();
        int resizeType = interpParam->resizeType();
        bool alignCorners = interpParam->alignCorners();
        bool halfPixelCenters = interpParam->halfPixelCenters();
        switch (interpParam->ctm()) {
            case CoordinateTransformationMode_AlignCorners:
                alignCorners = true;
                halfPixelCenters = false;
                break;
            case CoordinateTransformationMode_HalfPixels:
            case CoordinateTransformationMode_PytorchHalfPixels:
            case CoordinateTransformationMode_TensorflowHalfPixels:
                alignCorners = false;
                halfPixelCenters = true;
                break;
            case CoordinateTransformationMode_Asymmetric:
                alignCorners = false;
                halfPixelCenters = false;
                break;
            case CoordinateTransformationMode_NotSet:
            default:
                break;
        }
        if (resizeType == 2) {
            mNodeType = QNN_OP_RESIZE_BILINEAR;
            this->createParamScalar(QNN_OP_RESIZE_BILINEAR_PARAM_ALIGN_CORNERS,
                                    alignCorners);
            this->createParamScalar(
                QNN_OP_RESIZE_BILINEAR_PARAM_HALF_PIXEL_CENTERS,
                halfPixelCenters);
            this->createParamScalar(QNN_OP_RESIZE_BILINEAR_PARAM_ANTIALIAS,
                                    false);
        } else if (resizeType == 1 || resizeType == 4) {
            mNodeType = QNN_OP_RESIZE_NEAREST_NEIGHBOR;
            this->createParamScalar(
                QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_ALIGN_CORNERS,
                alignCorners);
            this->createParamScalar(
                QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_HALF_PIXEL_CENTERS,
                halfPixelCenters);
        } else {
            mNodeType = QNN_OP_RESIZE;
            const uint32_t interpolationMode =
                QNN_OP_RESIZE_INTERPOLATION_MODE_CUBIC;
            uint32_t transformationMode =
                QNN_OP_RESIZE_TRANSFORMATION_MODE_ASYMMETRIC;
            if (alignCorners) {
                transformationMode =
                    QNN_OP_RESIZE_TRANSFORMATION_MODE_ALIGN_CORNERS;
            } else if (halfPixelCenters) {
                transformationMode =
                    QNN_OP_RESIZE_TRANSFORMATION_MODE_HALF_PIXEL;
            }
            this->createParamScalar("interpolation_mode", interpolationMode);
            this->createParamScalar("transformation_mode", transformationMode);
            this->createParamScalar("exclude_outside", (uint32_t)0);
            this->createParamScalar("cubic_coeff", interpParam->cubicCoeffA());
        }
        this->addNodeCommon(inputs, outputs, 1);
        return NO_ERROR;
    }

    mParams.clear();
    mInputs.clear();
    mOutputs.clear();

    auto interpParam = mOp->main_as_Interp();
    int resizeType = interpParam->resizeType();
    bool alignCorners = interpParam->alignCorners();
    bool halfPixelCenters = interpParam->halfPixelCenters();

    // ONNX exporters can leave an identity Resize in the graph when the
    // requested output size already matches the input. On V66, the generic
    // ResizeBilinear kernel still scans the full tensor even though the only
    // observable work is fixed-point requantization. Convert implements the
    // same scale/offset conversion without interpolation and is substantially
    // cheaper for large image tensors.
    if (mBackend->isDspBackend() && mBackend->requiresQuantizedGraph() &&
        inputs[0]->shape() == outputs[0]->shape()) {
        mNodeType = "Convert";
        mInputs.push_back(*(mBackend->getNativeTensor(inputs[0])));
        mOutputs.push_back(*(mBackend->getNativeTensor(outputs[0])));
        MNN_PRINT(
            "MNN_QNN_V66_IDENTITY_RESIZE: node=%s lowered=Convert "
            "elements=%d\n",
            mNodeName.c_str(), outputs[0]->elementSize());
        mBackend->addNodeToGraph(mOpConfigVersion, mNodeName.c_str(),
                                 mPackageName.c_str(), mNodeType.c_str(),
                                 mParams, mInputs, mOutputs);
        return NO_ERROR;
    }

    // Newer MNN models store the ONNX coordinate transformation mode in ctm;
    // alignCorners/halfPixelCenters are only legacy compatibility fields. The
    // Models using PytorchHalfPixels have semantics equivalent to QNN's
    // half_pixel_centers when both spatial output dimensions are greater than
    // one.
    switch (interpParam->ctm()) {
        case CoordinateTransformationMode_NotSet:
            break;
        case CoordinateTransformationMode_AlignCorners:
            alignCorners = true;
            halfPixelCenters = false;
            break;
        case CoordinateTransformationMode_HalfPixels:
            alignCorners = false;
            halfPixelCenters = true;
            break;
        case CoordinateTransformationMode_PytorchHalfPixels:
            if (outputs[0]->height() <= 1 || outputs[0]->width() <= 1) {
                MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
            }
            alignCorners = false;
            halfPixelCenters = true;
            break;
        case CoordinateTransformationMode_Asymmetric:
            alignCorners = false;
            halfPixelCenters = false;
            break;
        default:
            MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
    }

    // QNN 2.37 HTP validates the legacy resize op names. The generic Resize
    // form is accepted by newer SDK headers but rejected by the V73 backend.
    if (resizeType == 2) {
        mNodeType = QNN_OP_RESIZE_BILINEAR;
        this->createParamScalar(QNN_OP_RESIZE_BILINEAR_PARAM_ALIGN_CORNERS, alignCorners);
        this->createParamScalar(QNN_OP_RESIZE_BILINEAR_PARAM_ANTIALIAS, false);
        this->createParamScalar(QNN_OP_RESIZE_BILINEAR_PARAM_HALF_PIXEL_CENTERS, halfPixelCenters);
    } else if (resizeType == 1 || resizeType == 4) {
        mNodeType = QNN_OP_RESIZE_NEAREST_NEIGHBOR;
        this->createParamScalar(QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_ALIGN_CORNERS, alignCorners);
        this->createParamScalar(QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_HALF_PIXEL_CENTERS, halfPixelCenters);
    } else {
        MNN_QNN_NOT_SUPPORT_SPECIAL_CASE;
    }

    for (const auto &param : mParamScalarWrappers) {
        mParams.push_back(*(param->getNativeParam()));
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

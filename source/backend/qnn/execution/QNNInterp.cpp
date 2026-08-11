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

ErrorCode QNNInterp::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto interpParam = mOp->main_as_Interp();
    int resizeType = interpParam->resizeType();
    bool alignCorners = interpParam->alignCorners();
    bool halfPixelCenters = interpParam->halfPixelCenters();

    // The ONNX/Torch coordinate-transformation mode is carried in the `ctm` field, not in the
    // alignCorners/halfPixelCenters bools -- the converter only sets those bools for the exact
    // strings "align_corners"/"half_pixel" (see tools/converter/source/onnx/ResizeOnnx.cpp). For
    // modes like pytorch_half_pixel the bools are both false, which would otherwise make QNN fall
    // back to ASYMMETRIC coordinates and produce large errors on down/up-sampling. Derive the
    // effective flags from ctm, keeping the bools as fallback when ctm is NotSet.
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
            break; // keep the alignCorners/halfPixelCenters bools as-is
    }

    // Use ResizeBilinear for bilinear, ResizeNearestNeighbor for nearest, Resize for others
    if (resizeType == 2) {
        // Bilinear: use ResizeBilinear op which is verified on V73 HTP
        mNodeType = QNN_OP_RESIZE_BILINEAR;
        this->createParamScalar(QNN_OP_RESIZE_BILINEAR_PARAM_ALIGN_CORNERS, (bool)alignCorners);
        this->createParamScalar(QNN_OP_RESIZE_BILINEAR_PARAM_HALF_PIXEL_CENTERS, (bool)halfPixelCenters);
        // antialias must be explicitly set for V73 HTP validation
        this->createParamScalar(QNN_OP_RESIZE_BILINEAR_PARAM_ANTIALIAS, (bool)false);
    } else if (resizeType == 1 || resizeType == 4) {
        // Nearest: use ResizeNearestNeighbor op
        mNodeType = QNN_OP_RESIZE_NEAREST_NEIGHBOR;
        this->createParamScalar(QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_ALIGN_CORNERS, (bool)alignCorners);
        this->createParamScalar(QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_HALF_PIXEL_CENTERS, (bool)halfPixelCenters);
    } else {
        // Cubic or other: use generic Resize op
        mNodeType = QNN_OP_RESIZE;
        uint32_t interpolationMode = QNN_OP_RESIZE_INTERPOLATION_MODE_CUBIC;
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
        float cubicCoeff = interpParam->cubicCoeffA();
        this->createParamScalar("cubic_coeff", cubicCoeff);
    }

    // ResizeBilinear/ResizeNearestNeighbor only takes 1 input (the image tensor)
    // MNN's Interp op may have a 2nd input (size tensor) which QNN doesn't need
    // Output shape is determined by the output tensor dimensions
    this->addNodeCommon(inputs, outputs, 1);

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
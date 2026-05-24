#include <stdio.h>
#include "onnxOpConverter.hpp"
#include "logkit.h"

DECLARE_OP_CONVERTER(ResizeOnnx);

MNN::OpType ResizeOnnx::opType() {
    return MNN::OpType_Interp;
}

MNN::OpParameter ResizeOnnx::type() {
    return MNN::OpParameter_Interp;
}

void ResizeOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode, OnnxScope* scope) {
    std::unique_ptr<MNN::InterpT> resizeParam(new MNN::InterpT);
    std::string resizeMode = "";
    std::string coordMode = "half_pixel";
    std::string nearestMode = "round_prefer_floor";
    float cubicFactor = -0.75f;
    for (int i = 0; i < onnxNode->attribute_size(); ++i) {
        const auto& attr = onnxNode->attribute(i);
        const auto& key = attr.name();
        if (key == "mode") {
            resizeMode = attr.s();
        } else if (key == "coordinate_transformation_mode") {
            coordMode = attr.s();
        } else if (key == "nearest_mode") {
            nearestMode = attr.s();
        } else if (key == "cubic_coeff_a") {
            cubicFactor = attr.f();
        }
    }

    if (resizeMode == "nearest") {
        if (nearestMode == "round_prefer_floor") {
            resizeParam->resizeType = 4;
        } else if (nearestMode == "floor") {
            resizeParam->resizeType = 1;
        } else {
            LOG(ERROR) << "Don't support " << nearestMode << " nearest mode, use round_prefer_floor instead";
            resizeParam->resizeType = 4;
        }
    } else if (resizeMode == "bilinear" || resizeMode == "linear") {
        resizeParam->resizeType = 2;
    } else if (resizeMode == "cubic") {
        resizeParam->resizeType = 3;
        resizeParam->cubicCoeffA = cubicFactor;
    } else {
        LOG(ERROR) << "Unsupported Resize mode " << resizeMode << ", use bilinear instead";
        resizeParam->resizeType = 2;
    }

    resizeParam->alignCorners = (coordMode == "align_corners");
    resizeParam->halfPixelCenters = (coordMode == "half_pixel");
#define SET_MODE(str, c)  \
    if (coordMode == str) \
    resizeParam->ctm = MNN::CoordinateTransformationMode_##c
    SET_MODE("align_corners", AlignCorners);
    SET_MODE("half_pixel", HalfPixels);
    SET_MODE("pytorch_half_pixel", PytorchHalfPixels);
    SET_MODE("tf_half_pixel_for_nn", TensorflowHalfPixels);
    SET_MODE("tf_crop_and_resize", TensorflowCropAndResize);
    SET_MODE("asymmetric", Asymmetric);
#undef SET_MODE

    // Treat an ONNX optional input as "absent" if its name is empty OR it points to a
    // named-but-empty initializer (shape produces zero elements) — both conventions are
    // valid ONNX. Otherwise fall back to whatever the producer emits (constant or dynamic).
    auto isOptionalInputProvided = [&](int idx) -> bool {
        if (idx >= onnxNode->input_size()) {
            return false;
        }
        const auto& name = onnxNode->input(idx);
        if (name.empty()) {
            return false;
        }
        auto iter = scope->mInitializers.find(name);
        if (iter == scope->mInitializers.end()) {
            return true;
        }
        int64_t numel = 1;
        for (int i = 0; i < iter->second->dims_size(); ++i) {
            numel *= iter->second->dims(i);
        }
        return numel > 0;
    };

    // ONNX opset 10 Resize has only (X, scales). Opset 11+ uses (X, roi, scales, sizes).
    // sizes (when given) takes precedence over scales.
    int shapeInputIdx = -1;
    if (isOptionalInputProvided(3)) {
        shapeInputIdx = 3;
    } else if (isOptionalInputProvided(2)) {
        shapeInputIdx = 2;
    } else if (onnxNode->input_size() == 2 && isOptionalInputProvided(1)) {
        shapeInputIdx = 1; // opset 10
    }

    std::vector<int> inputIndexes;
    auto dataIndex = scope->lookupTensor(onnxNode->input(0));
    if (dataIndex >= 0) {
        inputIndexes.emplace_back(dataIndex);
    }
    if (shapeInputIdx >= 0) {
        auto shapeIndex = scope->lookupTensor(onnxNode->input(shapeInputIdx));
        if (shapeIndex >= 0) {
            inputIndexes.emplace_back(shapeIndex);
        }
    }
    dstOp->inputIndexes = std::move(inputIndexes);

    dstOp->main.value = resizeParam.release();
    dstOp->defaultDimentionFormat = MNN::MNN_DATA_FORMAT_NCHW;
}

REGISTER_CONVERTER(ResizeOnnx, Resize);

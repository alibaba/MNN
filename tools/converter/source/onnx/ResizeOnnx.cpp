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

void ResizeOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                     OnnxScope* scope) {
    std::unique_ptr<MNN::InterpT> resizeParam(new MNN::InterpT);
    bool bakedScales = false;
    bool bakedSizes  = false;
    bool is3DResize  = false;
    std::string resizeMode  = "";
    std::string coordMode   = "half_pixel";
    std::string nearestMode = "round_prefer_floor";
    float cubicFactor       = -0.75f;
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

    resizeParam->alignCorners     = (coordMode == "align_corners");
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

    // If scales / sizes are constant, bake them into Interp and drop extra shape input.
    if (onnxNode->input_size() >= 3) {
        const auto& scalesName = onnxNode->input(2);
        auto iter = scope->mInitializers.find(scalesName);
        if (iter != scope->mInitializers.end()) {
            std::unique_ptr<MNN::OpT> tempOp(new MNN::OpT);
            std::unique_ptr<MNN::BlobT> blob(onnxOpConverter::convertTensorToBlob(iter->second, scope->mModelDir, tempOp.get()));
            if (blob && !blob->float32s.empty()) {
                auto size = (int)blob->float32s.size();
                if (size == 4 || size == 5) {
                    is3DResize = is3DResize || size == 5;
                    resizeParam->heightScale = blob->float32s[size - 2];
                    resizeParam->widthScale = blob->float32s[size - 1];
                    if (size == 5) {
                        resizeParam->depthScale = blob->float32s[2];
                    }
                    bakedScales = true;
                }
            }
        }
    }
    if (onnxNode->input_size() >= 4) {
        const auto& sizesName = onnxNode->input(3);
        auto iter = scope->mInitializers.find(sizesName);
        if (iter != scope->mInitializers.end()) {
            std::unique_ptr<MNN::OpT> tempOp(new MNN::OpT);
            std::unique_ptr<MNN::BlobT> blob(onnxOpConverter::convertTensorToBlob(iter->second, scope->mModelDir, tempOp.get()));
            if (blob && !blob->int32s.empty()) {
                auto size = (int)blob->int32s.size();
                if (size == 4 || size == 5) {
                    is3DResize = is3DResize || size == 5;
                    if (size == 5) {
                        resizeParam->outputDepth = blob->int32s[2];
                    }
                    resizeParam->outputHeight = blob->int32s[size - 2];
                    resizeParam->outputWidth = blob->int32s[size - 1];
                    bakedSizes = true;
                }
            }
        }
    }

    std::vector<int> inputIndexes;
    auto dataIndex = scope->lookupTensor(onnxNode->input(0));
    if (dataIndex >= 0) {
        inputIndexes.emplace_back(dataIndex);
    }
    if (!bakedSizes && onnxNode->input_size() >= 4 && !onnxNode->input(3).empty()) {
        auto sizesIndex = scope->lookupTensor(onnxNode->input(3));
        if (sizesIndex >= 0) {
            inputIndexes.emplace_back(sizesIndex);
        }
    } else if (!bakedScales && onnxNode->input_size() >= 3 && !onnxNode->input(2).empty()) {
        auto scalesIndex = scope->lookupTensor(onnxNode->input(2));
        if (scalesIndex >= 0) {
            inputIndexes.emplace_back(scalesIndex);
        }
    }
    dstOp->inputIndexes = std::move(inputIndexes);
    if (is3DResize) {
        dstOp->type = MNN::OpType_Interp3D;
    }

    dstOp->main.value = resizeParam.release();
    dstOp->defaultDimentionFormat = MNN::MNN_DATA_FORMAT_NCHW;
}

REGISTER_CONVERTER(ResizeOnnx, Resize);

//
//  OnnxUpsample.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

class OnnxUpSampleTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();

        std::vector<float> scales;
        int scalesSize = 1;

        auto op            = expr->get();
        auto extraParam    = op->main_as_Extra();
        const int attrSize = extraParam->attr()->size();
        std::string interpMode;
        std::string coordMode = ""; // detect align_corner attribute
        for (int i = 0; i < attrSize; ++i) {
            auto attr       = extraParam->attr()->GetAs<Attribute>(i);
            const auto& key = attr->key()->str();
            if (key == "mode") {
                interpMode = attr->s()->str();
            } else if ((inputs.size() == 1) && key == "scales") {
                scalesSize = attr->list()->f()->size();
                scales.resize(scalesSize);
                memcpy(scales.data(), attr->list()->f()->data(), sizeof(float) * scalesSize);
            } else if (key == "coordinate_transformation_mode") {
                coordMode = attr->s()->str();
            }
        }

        std::unique_ptr<OpT> mergeredUpsample(new OpT);
        mergeredUpsample->name      = expr->name();
        mergeredUpsample->type      = OpType_Interp;
        mergeredUpsample->main.type = OpParameter_Interp;

        std::unique_ptr<InterpT> interpParam(new InterpT);

        const float* scaleDataPtr = scales.data();

        if (inputs.size() == 2) {
            auto scale     = inputs[1];
            scaleDataPtr   = scale->readMap<float>();
            auto scaleInfo = scale->getInfo();

            if (!scaleDataPtr) {
                mergeredUpsample->main.value = interpParam.release();
                auto output = Variable::create(Expr::create(mergeredUpsample.get(), {inputs[0], inputs[1]}));
                return output->expr().first;
            }
            // scale is constant node
            scalesSize = scaleInfo->size;
        }

        interpParam->widthScale  = 1.0f;
        interpParam->heightScale = 1.0f;
        if (scalesSize >= 2 && scalesSize <= 4) {
            MNN_THROW_CHECK(scaleDataPtr[1] == 1.0f, "MNN NOT SUPPORT Upsamle along with channle");
            if (scalesSize >= 3) {
                interpParam->heightScale = scaleDataPtr[2];
            }
            if (scalesSize == 4){
                interpParam->widthScale  = scaleDataPtr[3];
            } 
        } else {
            MNN_ERROR("MNN Not support Upsample when scale size = %d\n", scalesSize);
        }
        interpParam->alignCorners = (coordMode == "align_corners");

        // 1:near 2: bilinear 3: cubic
        if (interpMode == "nearest") {
            interpParam->resizeType = 1;
        } else if (interpMode == "bilinear" || interpMode == "linear") {
            interpParam->resizeType = 2;
        } else if (interpMode == "cubic") {
            interpParam->resizeType = 3;
        } else {
            MNN_ERROR("Unsupported Upsample mode! ==> %s\n", interpMode.c_str());
        }

        mergeredUpsample->main.value = interpParam.release();
        auto newInput                = inputs[0];
        auto tempOutput              = Variable::create(Expr::create(mergeredUpsample.get(), {newInput}));
        tempOutput->setName(expr->name());

        auto output = tempOutput;
        return output->expr().first;
    }
};

class OnnxReiszeTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        // input, roi, scales, sizes
        // for more information, please reference from https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize
        MNN_THROW_CHECK((inputs.size() >= 2), "Onnx Resize should have 2/3/4 inputs!");
        std::string resizeMode  = "";
        std::string coordMode   = "half_pixel"; // detect align_corner attribute
        std::string nearestMode = "round_prefer_floor";
        auto op                 = expr->get();
        auto extraParam         = op->main_as_Extra();
        const int attrSize      = extraParam->attr()->size();
        float cubicFactor       = -0.75f;
        for (int i = 0; i < attrSize; ++i) {
            auto attr       = extraParam->attr()->GetAs<Attribute>(i);
            const auto& key = attr->key()->str();
            if (key == "mode") {
                resizeMode = attr->s()->str();
            } else if (key == "coordinate_transformation_mode") {
                coordMode = attr->s()->str();
            } else if (key == "nearest_mode") {
                nearestMode = attr->s()->str();
            } else if (key == "cubic_coeff_a") {
                cubicFactor = attr->f();
            }
        }

        std::unique_ptr<OpT> mergeredResize(new OpT);
        mergeredResize->type      = OpType_Interp;
        mergeredResize->main.type = OpParameter_Interp;

        std::unique_ptr<InterpT> resizeParam(new InterpT);
        // 1:near 2: bilinear 3: cubic
        if (resizeMode == "nearest") {
            if (nearestMode == "round_prefer_floor") {
                resizeParam->resizeType = 4;
            } else if (nearestMode == "floor") {
                resizeParam->resizeType = 1;
            } else {
                MNN_ERROR("Don't support %s neareset mode, use round_prefer_floor instead\n", nearestMode.c_str());
                resizeParam->resizeType = 4;
            }
        } else if (resizeMode == "bilinear" || resizeMode == "linear") {
            resizeParam->resizeType = 2;
        } else if (resizeMode == "cubic") {
            resizeParam->resizeType  = 3;
            resizeParam->cubicCoeffA = cubicFactor;
        } else {
            MNN_ERROR("Unsupported Upsample mode! ==> %s, use bilinear instead\n", resizeMode.c_str());
            resizeParam->resizeType = 2;
        }
        // For compability of old mnn
        resizeParam->alignCorners     = (coordMode == "align_corners");
        resizeParam->halfPixelCenters = (coordMode == "half_pixel");

        /*
        coordinate_transformation_mode: string
        This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original
        tensor.

        The coordinate of each dimension is transformed individually. Let's describe a case using axis x as an example.
        Denote x_resized as the coordinate of axis x in the resized tensor, x_original as the coordinate of axis x in
        the original tensor, length_original as the length of the original tensor in axis x, length_resized as the
        length of the resized tensor in axis x, roi_x = (start_x, end_x) of the axis x in input "roi", scale =
        length_resized / length_original,

        if coordinate_transformation_mode is "half_pixel",
        x_original = (x_resized + 0.5) / scale - 0.5,

        if coordinate_transformation_mode is "pytorch_half_pixel",
        x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0,

        if coordinate_transformation_mode is "align_corners",
        x_original = x_resized * (length_original - 1) / (length_resized - 1),

        if coordinate_transformation_mode is "asymmetric",
        x_original = x_resized / scale,

        if coordinate_transformation_mode is "tf_half_pixel_for_nn",
        x_original = (x_resized + 0.5) / scale,

        if coordinate_transformation_mode is "tf_crop_and_resize",
        x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) *
        (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1).
         */
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

        VARP output;
        if (inputs.size() == 2) {
            mergeredResize->main.value = resizeParam.release();
            auto output = Variable::create(Expr::create(mergeredResize.get(), {inputs[0], inputs[1]}));
            output->setName(expr->name());
            return output->expr().first;
        }
        if (inputs.size() == 3) {
            auto scaleT = inputs[2];
            // Compute shape dynamic
            mergeredResize->main.value = resizeParam.release();
            auto resizeExpr            = Expr::create(mergeredResize.get(), {inputs[0], {inputs[2]}});
            resizeExpr->setName(expr->name());
            output = Variable::create(resizeExpr);
            return output->expr().first;
        }
        if (inputs.size() == 4) {
            auto sizes                 = inputs[3];
            auto name                  = sizes->name();
            mergeredResize->main.value = resizeParam.release();
            auto resizeExpr            = Expr::create(mergeredResize.get(), {inputs[0], inputs[3]});
            resizeExpr->setName(expr->name());
            output = Variable::create(resizeExpr);
            return output->expr().first;
        }
        return output->expr().first;
    }
};

static auto gRigister = []() {
    OnnxExtraManager::get()->insert("Upsample",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxUpSampleTransform));

    OnnxExtraManager::get()->insert("Resize", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxReiszeTransform));
    return true;
}();

} // namespace Express
} // namespace MNN

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
                memcmp(scales.data(), attr->list()->f()->data(), sizeof(float) * scalesSize);
            } else if (key == "coordinate_transformation_mode") {
                coordMode = attr->s()->str();
            }
        }

        std::unique_ptr<OpT> mergeredUpsample(new OpT);
        mergeredUpsample->name = expr->name();
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
                auto output = Variable::create(Expr::create(mergeredUpsample.get(), {_Convert(inputs[0], NC4HW4), inputs[1]}));
                output = _Convert(output, NCHW);
                return output->expr().first;
            }
            // scale is constant node
            scalesSize = scaleInfo->size;
        }

        interpParam->widthScale  = 1.0f;
        interpParam->heightScale = 1.0f;
        if (scalesSize == 2) {
            interpParam->widthScale = scaleDataPtr[1];
        } else if (scalesSize == 3) {
            interpParam->widthScale  = scaleDataPtr[2];
            interpParam->heightScale = scaleDataPtr[1];
        } else if (scalesSize == 4) {
            interpParam->widthScale  = scaleDataPtr[3];
            interpParam->heightScale = scaleDataPtr[2];
            MNN_CHECK(scaleDataPtr[1] == 1.0f, "MNN NOT SUPPORT Upsamle along with channle");
        } else {
            MNN_ERROR("MNN Not support Upsample when scale size = %d\n", scalesSize);
        }
        interpParam->alignCorners = (coordMode == "align_corners");
        interpParam->halfPixelCenters = (interpParam->alignCorners == false);
        
        // 1:near 2: bilinear 3: cubic
        if (interpMode == "nearest") {
            interpParam->resizeType = 1;
        } else if (interpMode == "bilinear" || interpMode == "linear") {
            interpParam->resizeType = 2;
        } else {
            MNN_ERROR("Unsupported Upsample mode! ==> %s\n", interpMode.c_str());
        }

        mergeredUpsample->main.value = interpParam.release();
        auto newInput = _Convert(inputs[0], NC4HW4);
        auto tempOutput = Variable::create(Expr::create(mergeredUpsample.get(), {newInput}));
        tempOutput->setName(expr->name());

        auto output = _Convert(tempOutput, NCHW);
        return output->expr().first;
    }
};

class OnnxReiszeTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        // input, roi, scales, sizes
        // for more information, please reference from https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize
        MNN_CHECK((inputs.size() == 4) || (inputs.size() == 2), "Onnx Resize should have 4 or 2 inputs!");

        std::string resizeMode = "";
        std::string coordMode = ""; // detect align_corner attribute
        auto op                = expr->get();
        auto extraParam        = op->main_as_Extra();
        const int attrSize     = extraParam->attr()->size();
        for (int i = 0; i < attrSize; ++i) {
            auto attr       = extraParam->attr()->GetAs<Attribute>(i);
            const auto& key = attr->key()->str();
            if (key == "mode") {
                resizeMode = attr->s()->str();
            } else if (key == "coordinate_transformation_mode") {
                coordMode = attr->s()->str();
            }
        }

        std::unique_ptr<OpT> mergeredResize(new OpT);
        mergeredResize->type      = OpType_Interp;
        mergeredResize->main.type = OpParameter_Interp;

        std::unique_ptr<InterpT> resizeParam(new InterpT);
        // 1:near 2: bilinear 3: cubic
        if (resizeMode == "nearest") {
            resizeParam->resizeType = 1;
        } else if (resizeMode == "bilinear" || resizeMode == "linear") {
            resizeParam->resizeType = 2;
        } else {
            MNN_ERROR("Unsupported Upsample mode! ==> %s\n", resizeMode.c_str());
        }
        resizeParam->alignCorners = (coordMode == "align_corners");
        resizeParam->halfPixelCenters = (resizeParam->alignCorners == false);

        VARP output;
        if (inputs.size() == 2) {
            auto ptr = inputs[1]->readMap<float>();
            MNN_ASSERT((ptr[0] == 1) && (ptr[1] == 1));
            resizeParam->heightScale = ptr[2];
            resizeParam->widthScale = ptr[3];
            mergeredResize->main.value = resizeParam.release();
            auto resizeExpr = Expr::create(mergeredResize.get(), {_Convert(inputs[0], NC4HW4)});
            resizeExpr->setName(expr->name());
            output = _Convert(Variable::create(resizeExpr), NCHW);
            return output->expr().first;
        }

        auto sizes = inputs[3];
        auto name         = sizes->name();
        auto sizesDataPtr = sizes->readMap<int32_t>();
        if (!sizesDataPtr) {
            mergeredResize->main.value = resizeParam.release();
            auto resizeExpr = Expr::create(mergeredResize.get(), {_Convert(inputs[0], NC4HW4), inputs[2]});
            resizeExpr->setName(expr->name());
            output = _Convert(Variable::create(resizeExpr), NCHW);
        } else {
            auto scalesInfo      = sizes->getInfo();
            const int scalesSize = scalesInfo->size;
            MNN_CHECK(scalesSize == 4, "ONNX resize sizes should have 4 elements(n,c,h,w)");
            resizeParam->outputHeight = sizesDataPtr[2];
            resizeParam->outputWidth  = sizesDataPtr[3];

            mergeredResize->main.value = resizeParam.release();
            auto resizeExpr = Expr::create(mergeredResize.get(), {_Convert(inputs[0], NC4HW4)});
            resizeExpr->setName(expr->name());
            output = _Convert(Variable::create(resizeExpr), NCHW);
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

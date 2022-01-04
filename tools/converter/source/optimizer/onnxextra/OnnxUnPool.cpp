//
//  OnnxUnPool.cpp
//  MNNConverter
//
//  Created by MNN on 2020/01/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

/*
 MaxUnPool implemention is same as onnxruntime / pytorch
 Note: test case in onnx's docs is wrong. https://github.com/onnx/onnx/issues/2398
 */
class OnnxUnPoolTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        INTS kernels, pads, strides;
        auto attrs = expr->get()->main_as_Extra()->attr();
        auto extractVec = [](INTS& vec, const Attribute* attr) {
            vec.assign(attr->list()->i()->begin(), attr->list()->i()->end());
        };
        for (int i = 0; i < attrs->size(); ++i) {
            auto attr = attrs->GetAs<Attribute>(i);
            const auto& key = attr->key()->str();
            if (key == "kernel_shape") {
                extractVec(kernels, attr);
            } else if (key == "pads") {
                extractVec(pads, attr);
            } else if (key == "strides") {
                extractVec(strides, attr);
            }
        }
        auto inputs = expr->inputs();
        VARP outShape;
        if (inputs.size() == 3) {
            outShape = inputs[2];
        } else {
            int len = kernels.size();
            MNN_THROW_CHECK(strides.size() == 0 || strides.size() == len, "Invalid strides attr");
            MNN_THROW_CHECK(pads.size() == 0 || pads.size() == len * 2, "Invalid pads attr");
            auto kernelV = _Const(kernels.data(), {len}, NCHW, halide_type_of<int>());
            auto oneV = _Scalar<int>(1), zeroV = _Scalar<int>(0);
            auto strideV = oneV, padV = zeroV;
            if (strides.size() != 0) {
                strideV = _Const(strides.data(), {len}, NCHW, halide_type_of<int>());
            }
            if (pads.size() != 0) {
                INTS newPads;
                for (int i = 0; i < len; ++i) {
                    newPads.push_back(pads[i] + pads[i + len]);
                }
                padV = _Const(newPads.data(), {len}, NCHW, halide_type_of<int>());
            }
            auto inShape = _Shape(inputs[0], NCHW), twoV = _Unsqueeze(_Scalar<int>(2), {0});
            outShape = _Slice(inShape, twoV, _Unsqueeze(_Scalar<int>(len), {0}));
            outShape = kernelV + (outShape - oneV) * strideV - padV;
            outShape = _Concat({_Slice(inShape, _Unsqueeze(zeroV, {0}), twoV), outShape}, 0);
        }
        auto res = _ScatterNd(_Reshape(inputs[1], {-1, 1}), _Reshape(inputs[0], {-1}), _Unsqueeze(_ReduceProd(outShape), {0}));
        res = _Reshape(res, outShape);
        res->setName(expr->outputName(0));
        return res->expr().first;
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("MaxUnpool", std::shared_ptr<OnnxExtraManager::Transform>(new OnnxUnPoolTransform));
    return true;
}();

} // namespace Express
} // namespace MNN

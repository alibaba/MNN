//
//  OnnxConstantOfShape.cpp
//  MNNConverter
//
//  Created by MNN on 2019/10/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNN_generated.h"
#include "OnnxExtraManager.hpp"

namespace MNN {
namespace Express {

template <typename T>
static void ResizeAndCopyData(std::vector<uint8_t> *vec_data, const T *src, const std::vector<int> &shape) {
    int64_t count = sizeof(T);
    for (const auto &d : shape) {
        count *= d;
    }
    MNN_ASSERT(count > 0);

    vec_data->resize(count);
    memcpy(vec_data->data(), src, count);
}

class OnnxConstantOfShapeTransform : public OnnxExtraManager::Transform {
public:
    virtual EXPRP onExecute(EXPRP expr) const override {
        auto inputs = expr->inputs();
        MNN_THROW_CHECK(1 == inputs.size(), "Onnx ConstantOfShape should have one input!");

        std::unique_ptr<OpT> mnnFill(new OpT);
        mnnFill->name                   = expr->name();
        mnnFill->defaultDimentionFormat = MNN_DATA_FORMAT_NCHW;
        mnnFill->type                   = OpType_Fill;
        mnnFill->main.type              = OpParameter_NONE;
        mnnFill->main.value             = nullptr;

        std::vector<uint8_t> tensor_data;
        std::vector<int> tensor_shape;
        halide_type_t data_type = halide_type_of<float>();
        auto extraParam         = expr->get()->main_as_Extra();
        VARP const_shape;
        if (extraParam->attr() != nullptr) {
            for (int i = 0; i < extraParam->attr()->size(); ++i) {
                auto attr       = extraParam->attr()->GetAs<Attribute>(i);
                const auto &key = attr->key()->str();
                if (key == "value") {
                    auto blob = attr->tensor();
                    // Process tensor shape.
                    tensor_shape.resize(blob->dims()->size());
                    for (int j = 0; j < tensor_shape.size(); ++j) {
                        tensor_shape[j] = blob->dims()->Get(j);
                    }

                    if (blob->dataType() == DataType_DT_INT32) {
                        data_type = halide_type_of<int32_t>();
                        ResizeAndCopyData<int>(&tensor_data, blob->int32s()->data(), tensor_shape);
                    } else if (blob->dataType() == DataType_DT_FLOAT) {
                        data_type = halide_type_of<float>();
                        ResizeAndCopyData<float>(&tensor_data, blob->float32s()->data(), tensor_shape);
                    } else {
                        MNN_ERROR("Not support data type.");
                    }
                    break; // Break out of the loop if value has been processed.
                }
            }
            const_shape = _Const(static_cast<const void *>(tensor_data.data()), tensor_shape, NCHW, data_type);
        } else {
            // https://github.com/onnx/onnx/blob/master/docs/Operators.md#attributes-12
            const_shape = _Scalar<float>(0.0f);
        }
        return Expr::create(mnnFill.get(), {inputs[0], const_shape});
    }
};

static auto gRegister = []() {
    OnnxExtraManager::get()->insert("ConstantOfShape",
                                    std::shared_ptr<OnnxExtraManager::Transform>(new OnnxConstantOfShapeTransform));
    return true;
}();

} // namespace Express
} // namespace MNN

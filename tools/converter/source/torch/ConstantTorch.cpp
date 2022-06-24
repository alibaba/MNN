//
//  ConstantTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(ConstantTorch);

MNN::OpType ConstantTorch::opType() {
    return MNN::OpType_Const;
}
MNN::OpParameter ConstantTorch::type() {
    return MNN::OpParameter_Blob;
}
std::vector<int> ConstantTorch::inputTensorIdx() {
    return {};
}

void ConstantTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, TorchScope* scope) {
    auto param = new MNN::BlobT;
    const auto output = node->output();
    const std::string& type = output->type()->str();
    if (type == "None") {
        return;
    }
    if (getRealOpType(node) == "Uninitialized" || node->attributeNames().empty()) {
        param->dataType = MNN::DataType_DT_FLOAT;
        param->float32s = {};
        dstOp->main.value = param;
        return;
    }
    auto attr = node->attributeNames()[0];
    auto kind = node->kindOf(attr);
    switch (kind) {
        case torch::jit::AttributeKind::f:
            param->dataType = MNN::DataType_DT_FLOAT;
            param->float32s.push_back(node->f(attr));
            break;
        case torch::jit::AttributeKind::i:
            param->dataType = MNN::DataType_DT_INT32;
            // node->i is int64_t
            param->int32s.push_back(std::min(node->i(attr), static_cast<int64_t>(std::numeric_limits<int>::max())));
            break;
        case torch::jit::AttributeKind::s:
            param->dataType = MNN::DataType_DT_STRING;
            param->strings.push_back(node->s(attr));
            break;
        case torch::jit::AttributeKind::ival: {
            param->dataType = MNN::DataType_DT_INT32;
            const auto int64s = getValue<std::vector<int64_t>>(output);
            param->int32s.resize(int64s.size());
            param->dims.push_back(int64s.size());
            for (int i = 0; i < int64s.size(); i++) {
                param->int32s[i] = int64s[i];
            }
            break;
        }
        case torch::jit::AttributeKind::t: {
            const auto tensor = getValue<at::Tensor>(output);
            auto scalarType = tensor.scalar_type();
            switch (scalarType) {
                case at::ScalarType::Byte:
                    param->dataType = MNN::DataType_DT_UINT8;
                    param->uint8s = std::move(getValue<uint8_t>(output, param->dims));
                    break;
                case at::ScalarType::Char:
                    param->dataType = MNN::DataType_DT_INT8;
                    param->int8s = std::move(getValue<int8_t>(output, param->dims));
                    break;
                case at::ScalarType::Int:
                    param->dataType = MNN::DataType_DT_INT32;
                    param->int32s = std::move(getValue<int32_t>(output, param->dims));
                    break;
                case at::ScalarType::Long: {
                    param->dataType = MNN::DataType_DT_INT32;
                    const auto int64s = std::move(getValue<int64_t>(output, param->dims));
                    param->int32s.resize(int64s.size());
                    for (int i = 0; i < int64s.size(); i++) {
                        param->int32s[i] = int64s[i];
                    }
                    break;
                }
                case at::ScalarType::Float:
                    param->dataType = MNN::DataType_DT_FLOAT;
                    param->float32s = std::move(getValue<float>(output, param->dims));
                    break;
                case at::ScalarType::Double: {
                    param->dataType = MNN::DataType_DT_FLOAT;
                    const auto doubles = getValue<double>(output, param->dims);
                    param->float32s.resize(doubles.size());
                    for (int i = 0; i < doubles.size(); i++) {
                        param->float32s[i] = doubles[i];
                    }
                    break;
                }
                case at::ScalarType::Bool:
                    param->dataType = MNN::DataType_DT_INT32;
                    param->int32s = std::move(getValue<int32_t>(output, param->dims));
                    if (param->dims.empty() && param->int32s.empty()) {
                        param->int32s.push_back(0);
                        param->dims.push_back(1);
                    }
                    break;
                case at::ScalarType::BFloat16:
                case at::ScalarType::Short:
                case at::ScalarType::Half:
                default:
                    MNN_ASSERT(false);
                    break;
            }
            break;
        }
        default:
            MNN_ASSERT(false);
            return;
    }
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ConstantTorch, Constant);
REGISTER_CONVERTER(ConstantTorch, Uninitialized);

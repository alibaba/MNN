//
//  Utils.cpp
//  MNN
//
//  Created by MNN on 2019/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Utils.hpp"
#include <map>
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include "core/MNNMemoryUtils.h"
#include "core/Backend.hpp"
#include "core/Execution.hpp"

namespace MNN {
namespace Express {
Expr::Inside::Inside(int outputSize) {
    mOutputInfos.resize(outputSize);
    mOutputTensors.resize(outputSize);
    for (int i=0; i<outputSize; ++i) {
        mOutputTensors[i] = new Tensor;
        TensorUtils::getDescribe(mOutputTensors[i])->memoryType = Tensor::InsideDescribe::MEMORY_HOST;
    }
}
Expr::Inside::~Inside() {
    for (auto t : mOutputTensors) {
        delete t;
    }
}


#define CONVERT(src, dst, f)\
if (f == src) return dst;

int Utils::convertFormat(Dimensionformat format) {
    CONVERT(NCHW, MNN_DATA_FORMAT_NCHW, format);
    CONVERT(NHWC, MNN_DATA_FORMAT_NHWC, format);
    CONVERT(NC4HW4, MNN_DATA_FORMAT_NC4HW4, format);
    return MNN_DATA_FORMAT_UNKNOWN;
}

DataType Utils::convertDataType(halide_type_t type) {
    if (type.code == halide_type_float) {
        return DataType_DT_FLOAT;
    }
    if (type.code == halide_type_uint && type.bits == 8) {
        return DataType_DT_UINT8;
    }
    if (type.code == halide_type_int && type.bits == 8) {
        return DataType_DT_INT8;
    }
    if (type.code == halide_type_int && type.bits == 32) {
        return DataType_DT_INT32;
    }
    return DataType_DT_INVALID;
}
halide_type_t Utils::revertDataType(DataType dataType) {
    CONVERT(DataType_DT_FLOAT, halide_type_of<float>(), dataType);
    CONVERT(DataType_DT_INT32, halide_type_of<int32_t>(), dataType);
    CONVERT(DataType_DT_INT64, halide_type_of<int32_t>(), dataType);
    CONVERT(DataType_DT_UINT8, halide_type_of<uint8_t>(), dataType);
    CONVERT(DataType_DT_INT8, halide_type_of<int8_t>(), dataType);
    return halide_type_of<float>();
}
Express::Dimensionformat Utils::revertFormat(int format) {
    CONVERT(MNN_DATA_FORMAT_NCHW, Express::NCHW, format);
    CONVERT(MNN_DATA_FORMAT_NHWC, Express::NHWC, format);
    CONVERT(MNN_DATA_FORMAT_NC4HW4, Express::NC4HW4, format);
    return NCHW;
}
void Utils::copyInfoToTensor(Tensor* dest, const Variable::Info* source) {
    if (nullptr == source) {
        dest->buffer().dimensions = 0;
        return;
    }
    for (int i = 0; i < source->dim.size(); ++i) {
        dest->setLength(i, source->dim[i]);
    }
    dest->buffer().dimensions                       = (int)source->dim.size();
    dest->buffer().type                             = source->type;
    TensorUtils::getDescribe(dest)->dimensionFormat = (MNN_DATA_FORMAT)Utils::convertFormat(source->order);
    TensorUtils::setLinearLayout(dest);
}
void Utils::copyTensorToInfo(Variable::Info* shape, const Tensor* tensor) {
    shape->type  = tensor->getType();
    shape->dim   = tensor->shape();
    shape->size  = tensor->elementSize();
    shape->order = Utils::revertFormat(TensorUtils::getDescribe(tensor)->dimensionFormat);
}
bool Utils::allocMemoryForHostTensor(Tensor* dest) {
    if (nullptr != dest->buffer().host) {
        return true;
    }
    if (TensorUtils::getDescribe(dest)->memoryType != Tensor::InsideDescribe::MEMORY_HOST) {
        return false;
    }
    auto size = dest->size();
    if (0 >= size) {
        return false;
    }
    dest->buffer().host = (uint8_t*)MNNMemoryAllocAlign(size, MNN_MEMORY_ALIGN_DEFAULT);
    return dest->buffer().host != nullptr;
}
bool Utils::releaseMemoryForHostTensor(Tensor* dest) {
    if (nullptr == dest->buffer().host) {
        return true;
    }
    if (TensorUtils::getDescribe(dest)->memoryType != Tensor::InsideDescribe::MEMORY_HOST) {
        return false;
    }
    MNNMemoryFreeAlign(dest->buffer().host);
    dest->buffer().host = nullptr;
    return true;
}

template <>
void RearrangeWeights<MNN::OpType_Convolution>(Backend* backend,   // NOLINT
                                               const MNN::Op* op,  // NOLINT
                                               MNN::OpT* op_table) {
    Convolution2DT* conv_params = op_table->main.AsConvolution2D();
    // Return if the weights have been rearranged.
    if (conv_params->rearrangedParam &&  // NOLINT
        conv_params->rearrangedParam->type != RearrangedType_RT_NONE) {
        MNN_CHECK(conv_params->rearrangedParam->backend == backend->type(),
                  "Backend types are not match.");
        return;
    }
    std::unique_ptr<Execution> execution(
        backend->onCreate(std::vector<Tensor*>{}, std::vector<Tensor*>{}, op));
    std::vector<MNN::RearrangedType> types = execution->RearrangedTypes();
    std::vector<std::shared_ptr<Tensor>> weights =  // NOLINT
        execution->RearrangedWeights();

    if (types.empty() || weights.empty()) { return; }

    conv_params->weight.clear();
    conv_params->rearrangedParam.reset(new MNN::RearrangedWeightParamT);
    conv_params->rearrangedParam->backend = (int)backend->type();
    conv_params->rearrangedParam->type = types.at(0);
    conv_params->rearrangedParam->weight.resize(weights.at(0)->elementSize());
    memcpy(conv_params->rearrangedParam->weight.data(),  // NOLINT
           weights.at(0)->host<void>(), weights.at(0)->size());
}

template void RearrangeWeights<MNN::OpType_Convolution>(  // NOLINT
    Backend*, const MNN::Op*, MNN::OpT*);

} // namespace Express
} // namespace MNN

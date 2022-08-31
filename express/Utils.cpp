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
#include "core/ConvolutionCommon.hpp"

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
Expr::Inside::Inside(Tensor* tensor, bool own) {
    mOutputInfos.resize(1);
    mOutputTensors.resize(1);
    mOutputTensors[0] = tensor;
    Utils::copyTensorToInfo(&mOutputInfos[0], tensor);
    mOutputInfos[0].syncSize();
    mOwnTensor = own;
}

Expr::Inside::~Inside() {
    if (mOwnTensor) {
        for (auto t : mOutputTensors) {
            delete t;
        }
    }
    if (nullptr != mHostTensor) {
        delete mHostTensor;
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
    CONVERT(DataType_DT_HALF, halide_type_of<float>(), dataType);
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
Tensor* Utils::getTensor(VARP var) {
    auto exprInfo    = var->expr();
    auto inside      = exprInfo.first->inside();
    auto inputTensor = inside->mOutputTensors[exprInfo.second];
    if (nullptr != inside->mCache) {
        inputTensor = Executor::getOutput(inside->mCache.get(), inside->mCacheOffset);
    }
    return inputTensor;
}


} // namespace Express
} // namespace MNN

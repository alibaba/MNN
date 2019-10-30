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
#include "TensorUtils.hpp"
namespace MNN {
namespace Express {
int Utils::convertFormat(Dimensionformat format) {
    static std::map<Dimensionformat, MNN_DATA_FORMAT> gMap = {
        {NCHW, MNN_DATA_FORMAT_NCHW}, {NHWC, MNN_DATA_FORMAT_NHWC}, {NC4HW4, MNN_DATA_FORMAT_NC4HW4}};
    return gMap[format];
}

int Utils::convertDataType(halide_type_t type) {
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
static Express::Dimensionformat _convertFormat(MNN_DATA_FORMAT format) {
    static std::map<MNN_DATA_FORMAT, Express::Dimensionformat> gMap = {
        {MNN_DATA_FORMAT_NCHW, Express::NCHW},
        {MNN_DATA_FORMAT_NHWC, Express::NHWC},
        {MNN_DATA_FORMAT_NC4HW4, Express::NC4HW4},
    };
    return gMap[format];
}
static MNN_DATA_FORMAT _revertFormat(Express::Dimensionformat format) {
    static std::map<Express::Dimensionformat, MNN_DATA_FORMAT> gRevertMap = {
        {Express::NCHW, MNN_DATA_FORMAT_NCHW},
        {Express::NHWC, MNN_DATA_FORMAT_NHWC},
        {Express::NC4HW4, MNN_DATA_FORMAT_NC4HW4},
    };
    return gRevertMap[format];
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
    dest->buffer().host                             = (uint8_t*)source->ptr;
    TensorUtils::getDescribe(dest)->dimensionFormat = _revertFormat(source->order);
    TensorUtils::setLinearLayout(dest);
}
void Utils::copyTensorToInfo(Variable::Info* shape, const Tensor* tensor) {
    shape->type  = tensor->getType();
    shape->dim   = tensor->shape();
    shape->size  = tensor->elementSize();
    shape->order = _convertFormat(TensorUtils::getDescribe(tensor)->dimensionFormat);
    shape->ptr   = tensor->host<float>();
}
} // namespace Express
} // namespace MNN

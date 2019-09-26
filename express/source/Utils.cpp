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

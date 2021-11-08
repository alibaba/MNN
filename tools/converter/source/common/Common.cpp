//
//  Common.cpp
//  MNNConverter
//
//  Created by MNN on 2020/07/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Common.hpp"

namespace MNN {

MNN_DATA_FORMAT convertFormat(Express::Dimensionformat format) {
    switch (format) {
        case Express::NCHW:
            return MNN_DATA_FORMAT_NCHW;
        case Express::NHWC:
            return MNN_DATA_FORMAT_NHWC;
        case Express::NC4HW4:
            return MNN_DATA_FORMAT_NC4HW4;
        default:
            return MNN_DATA_FORMAT_UNKNOWN;
    }
}

DataType convertDataType(halide_type_t type) {
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

}  // namespace MNN

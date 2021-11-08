//
//  Common.hpp
//  MNNConverter
//
//  Created by MNN on 2020/07/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_CONVERTER_COMMON_COMMON_HPP_
#define MNN_CONVERTER_COMMON_COMMON_HPP_

#include "MNN/HalideRuntime.h"
#include "MNN/expr/Expr.hpp"

#include "MNN_generated.h"

namespace MNN {

DataType convertDataType(halide_type_t type);

MNN_DATA_FORMAT convertFormat(Express::Dimensionformat format);

}  // namespace MNN

#endif  // MNN_CONVERTER_COMMON_COMMON_HPP_

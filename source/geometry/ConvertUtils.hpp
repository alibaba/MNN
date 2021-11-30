//
//  ConvertUtils.hpp
//  MNN
//
//  Created by MNN on 2020/04/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvertUtils_hpp
#define ConvertUtils_hpp
#include "geometry/GeometryComputer.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
class ConvertUtils {
public:
    static bool compute(Tensor* input, Tensor* output, CommandBuffer& res);
    // numpy broadcast like: [3, 4] -> [2, 3, 4]
    // forward = true: [4] -> [4, 3, 2]
    static void broadcastto(Tensor* input, Tensor* output, bool forward = false);
};
} // namespace MNN

#endif

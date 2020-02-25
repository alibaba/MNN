//
//  Example.hpp
//  MNN
//
//  Created by MNN on 2019/11/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Example_hpp
#define Example_hpp

#include <MNN/MNNDefine.h>
#include <MNN/expr/ExprCreator.hpp>
#include <vector>

using namespace MNN::Express;

namespace MNN {
namespace Train {

class MNN_PUBLIC Example {
public:
    std::vector<VARP> data, target;

    Example() = default;
    Example(std::vector<VARP> data, std::vector<VARP> label) : data(std::move(data)), target(std::move(label)) {
    }
};

// class MNN_PUBLIC TensorExample {
// public:
//     VARP data;
//
//     TensorExample(VARP data) : data(std::move(data)) {}
// };

} // namespace Train
} // namespace MNN

#endif /* Example_hpp */

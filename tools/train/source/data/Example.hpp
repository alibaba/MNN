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
/**
 First: data: a vector of input tensors (for single input dataset is only one)
 Second: target: a vector of output tensors (for single output dataset is only one)
 */
typedef std::pair<std::vector<VARP>, std::vector<VARP>> Example;

} // namespace Train
} // namespace MNN

#endif /* Example_hpp */

//
//  MobilenetUtils.hpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MobilenetUtils_hpp
#define MobilenetUtils_hpp

namespace MNN {
namespace Train {
namespace Model {

// https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
int makeDivisible(int v, int divisor = 8, int minValue = 0);

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // MobilenetUtils_hpp

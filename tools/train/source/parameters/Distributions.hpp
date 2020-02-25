//
//  Distributions.hpp
//  MNN
//
//  Created by MNN on 2019/11/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Distributions_hpp
#define Distributions_hpp

#include <MNN/MNNDefine.h>
#include <random>

namespace MNN {
namespace Train {

class Distributions {
public:
    static void uniform(const int count, const float min, const float max, float* r, std::mt19937 gen);
    static void gaussian(const int count, const float mu, const float sigma, float* r, std::mt19937 gen);
};

} // namespace Train
} // namespace MNN

#endif // Distritutions_hpp

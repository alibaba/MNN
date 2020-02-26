//
//  Distributions.cpp
//  MNN
//
//  Created by MNN on 2019/11/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Distributions.hpp"
#include <cmath>

namespace MNN {
namespace Train {

void Distributions::uniform(const int count, const float min, const float max, float *r, std::mt19937 gen) {
    std::uniform_real_distribution<float> dis(min, std::nextafter(max, std::numeric_limits<float>::max()));
    for (int i = 0; i < count; i++) {
        r[i] = dis(gen);
    }
}

void Distributions::gaussian(const int count, const float mu, const float sigma, float *r, std::mt19937 gen) {
    std::normal_distribution<float> dis(mu, sigma);
    for (int i = 0; i < count; i++) {
        r[i] = dis(gen);
    }
}

} // namespace Train
} // namespace MNN

//
//  ADAM.hpp
//  MNN
//
//  Created by MNN on 2019/12/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ADAM_hpp
#define ADAM_hpp

#include <set>
#include <string>
#include <vector>
#include "ParameterOptimizer.hpp"
#include "SGD.hpp"

namespace MNN {
namespace Train {

class MNN_PUBLIC ADAM : public SGD {
public:
    void onAppend(const std::set<Express::VARP>& parameters) override;

    void onRemove(const std::set<Express::VARP>& parameters) override;

    virtual Express::VARP onComputeUpdateValue(Express::VARP param, Express::VARP grad) override;

    void setMomentum2(float momentum2);

    void setEps(float eps);

private:
    float mMomentum2 = 0.999; // default 0.999
    float mEps       = 1e-8;
    std::map<MNN::Express::VARP, MNN::Express::VARP> mHistory2;
};

} // namespace Train
} // namespace MNN

#endif // ADAM_hpp

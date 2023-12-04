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
    ADAM(std::shared_ptr<Express::Module> module);
    virtual ~ ADAM() = default;

    virtual Express::VARP onComputeUpdateValue(Express::VARP param, Express::VARP grad) override;
    virtual std::pair<std::vector<Express::VARP>, std::vector<Express::VARP>>  onMakeParameterUpdateGraphByGrad(const std::vector<ParameterOptGrad>& parameterGrads) override;

    float getMomentum2();

    void setMomentum2(float momentum2);

    float getEps();

    void setEps(float eps);

private:
    float mMomentum2 = 0.999; // default 0.999
    float mEps       = 1e-8;
    std::map<MNN::Express::VARP, MNN::Express::VARP> mHistory2;
};

} // namespace Train
} // namespace MNN

#endif // ADAM_hpp

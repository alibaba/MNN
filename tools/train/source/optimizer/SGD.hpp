//
//  SGD.hpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SGD_hpp
#define SGD_hpp

#include <MNN/expr/ExprCreator.hpp>
#include <string>
#include <vector>
#include "ParameterOptimizer.hpp"

namespace MNN {
namespace Train {

class MNN_PUBLIC SGD : public ParameterOptimizer {
public:
    SGD(std::shared_ptr<Express::Module> module);
    virtual ~ SGD() = default;
    virtual std::map<Express::VARP, Express::VARP> onGetNextParameter(Express::VARP loss) override;
    virtual std::pair<std::vector<Express::VARP>, std::vector<Express::VARP>>  onMakeParameterUpdateGraphByGrad(const std::vector<ParameterOptGrad>& parameterGrads) override;

    Express::VARP regularizeParameters(Express::VARP param, Express::VARP grad);

    virtual Express::VARP onComputeUpdateValue(Express::VARP param, Express::VARP grad);

    void setLearningRate(float rate);

    float getMomentum();

    void setMomentum(float momentum);

    float getWeightDecay();

    void setWeightDecay(float decay);

    RegularizationMethod getRegularizationMethod();

    void setRegularizationMethod(RegularizationMethod method);

    float currentLearningRate();

    void setGradBlockName(std::vector<std::string> block) {
        mGradBlockExprName = block;
    }

protected:
    float mLearningRate                        = 0.001f;
    float mMomentum                            = 0;
    float mWeightDecay                         = 0;
    RegularizationMethod mRegularizationMethod = L2;
    std::map<MNN::Express::VARP, MNN::Express::VARP> mHistory;

    // For Cache
    const Express::Expr* mLoss = nullptr;
    int mLossFromIndex         = 0;
    std::vector<std::string> mGradBlockExprName;
};

} // namespace Train
} // namespace MNN

#endif // SGD_hpp

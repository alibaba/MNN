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
#include <set>
#include <string>
#include <vector>
#include "ParameterOptimizer.hpp"

namespace MNN {
namespace Train {

class MNN_PUBLIC SGD : public ParameterOptimizer {
public:
    enum RegularizationMethod {
        L1,
        L2,
    };

    virtual std::map<Express::VARP, Express::VARP> onGetNextParameter(Express::VARP loss) override;

    Express::VARP regularizeParameters(Express::VARP param, Express::VARP grad);

    virtual Express::VARP onComputeUpdateValue(Express::VARP param, Express::VARP grad);

    void setLearningRate(float rate);

    void setMomentum(float momentum);

    void setWeightDecay(float decay);

    void setRegularizationMethod(RegularizationMethod method);

    float currentLearningRate();

    virtual void onAppend(const std::set<Express::VARP>& parameters) override;

    virtual void onRemove(const std::set<Express::VARP>& parameters) override;

protected:
    float mLearningRate                        = 0.001f;
    float mMomentum                            = 0;
    float mWeightDecay                         = 0;
    RegularizationMethod mRegularizationMethod = L2;
    std::map<MNN::Express::VARP, MNN::Express::VARP> mHistory;

    // For Cache
    const Express::Expr* mLoss = nullptr;
    int mLossFromIndex         = 0;
};

} // namespace Train
} // namespace MNN

#endif // SGD_hpp

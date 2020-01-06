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

    virtual Express::VARP regularizeParameters(Express::VARP param, Express::VARP grad);

    virtual Express::VARP computeUpdateValue(Express::VARP param, Express::VARP grad);

    void setLearningRate(float rate);

    void setMomentum(float momentum);

    void setWeightDecay(float decay);

    void setRegularizationMethod(RegularizationMethod method);

    void append(const std::set<Express::VARP>& parameters);

    void remove(const std::set<Express::VARP>& parameters);

    const std::set<Express::VARP>& parameters() const;

protected:
    float mLearningRate                        = 0.001f;
    float mMomentum                            = 0;
    float mWeightDecay                         = 0;
    RegularizationMethod mRegularizationMethod = L2;
    std::set<Express::VARP> mParameters;
    std::map<MNN::Express::VARP, MNN::Express::VARP> mHistory;
    int mStep = 0;

    // For Cache
    const Express::Expr* mLoss = nullptr;
    int mLossFromIndex         = 0;
};

} // namespace Train
} // namespace MNN

#endif // SGD_hpp

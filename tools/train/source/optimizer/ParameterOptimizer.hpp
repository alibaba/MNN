//
//  ParameterOptimizer.hpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ParameterOptimizer_hpp
#define ParameterOptimizer_hpp
#include <MNN/expr/Expr.hpp>
#include <set>
namespace MNN {
namespace Train {

class MNN_PUBLIC ParameterOptimizer {
public:
    ParameterOptimizer()          = default;
    virtual ~ParameterOptimizer() = default;
    bool step(Express::VARP loss);
    int currentStep();
    void setCurrentStep(int step);
    void append(const std::set<Express::VARP>& parameters);
    void remove(const std::set<Express::VARP>& parameters);

    virtual std::map<Express::VARP, Express::VARP> onGetNextParameter(Express::VARP loss) = 0;
    const std::set<Express::VARP>& parameters() const;

private:
    virtual void onAppend(const std::set<Express::VARP>& parameters) = 0;
    virtual void onRemove(const std::set<Express::VARP>& parameters) = 0;
    std::set<Express::VARP> mParameters;
    int mStep = 0;
};

} // namespace Train
} // namespace MNN

#endif

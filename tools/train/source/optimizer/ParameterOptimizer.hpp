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
#include <MNN/expr/Module.hpp>
#include <set>
namespace MNN {
namespace Train {
class MNN_PUBLIC ParameterOptimizer {
public:
    enum RegularizationMethod {
        L1,
        L2,
        L1L2,
    };

    ParameterOptimizer(std::shared_ptr<Express::Module> module);
    virtual ~ParameterOptimizer() = default;
    bool step(Express::VARP loss);
    int currentStep();
    void setCurrentStep(int step);
    static void makeLoopModel(const char* mnnFileName, std::vector<Express::VARP> outputs, const std::pair<std::vector<Express::VARP>, std::vector<Express::VARP>>& parameters);
    
    struct ParameterOptGrad {
        Express::VARP parameter;
        Express::VARP parameterGrad;
        Express::VARP learningRate;
    };

    virtual std::map<Express::VARP, Express::VARP> onGetNextParameter(Express::VARP loss) = 0;
    virtual std::pair<std::vector<Express::VARP>, std::vector<Express::VARP>>  onMakeParameterUpdateGraphByGrad(const std::vector<ParameterOptGrad>& parameterGrads);
    std::pair<std::vector<Express::VARP>, std::vector<Express::VARP>> makeParameterUpdateGraphByGrad(const std::vector<Express::VARP>& p, const std::vector<Express::VARP>& pd, const std::vector<Express::VARP>& lr);

    static ParameterOptimizer* createSGD(std::shared_ptr<Express::Module> module, float lr, float momentum, float weightDecay, RegularizationMethod method);
    static ParameterOptimizer* createADAM(std::shared_ptr<Express::Module> module, float lr, float momentum, float momentum2, float weightDecay, float eps, RegularizationMethod method);
protected:
    const std::set<Express::VARP>& trainable() const {
        return mTrainable;
    }
    std::shared_ptr<Express::Module> module() const {
        return mModule;
    }
private:
    int mStep = 0;
    std::shared_ptr<Express::Module> mModule;
    std::set<Express::VARP> mTrainable;
};

} // namespace Train
} // namespace MNN

#endif

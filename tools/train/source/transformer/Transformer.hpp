//
//  Transformer.hpp
//  MNN
//
//  Created by MNN on 2019/12/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Transformer_hpp
#define Transformer_hpp
#include <MNN/expr/Optimizer.hpp>
#include "OpConverter.hpp"
#include <MNN_generated.h>

namespace MNN {
namespace Train {
class MNN_PUBLIC Transformer {
public:
    struct TrainConfig {
        std::vector<std::string> noUpdateOps;
        std::vector<std::string> onlyUpdateOps;
        std::map<std::string, std::map<std::string, MNN::AttributeT*>> extraParams;
    };

    static std::shared_ptr<Express::Optimizer> turnModelToTrainable(TrainConfig config);
    static std::shared_ptr<Express::Optimizer> turnModelToInfer();
};

class MNN_PUBLIC TurnTrainable : public Express::Optimizer {
public:
    TurnTrainable(Transformer::TrainConfig config) {
        mConfig = std::move(config);
    }
    virtual Cost onMeasure(const std::vector<Express::VARP>& outputs,
                           std::shared_ptr<Parameters> parameters = nullptr) override {
        return Cost();
    }
    virtual bool onExecute(const std::vector<Express::VARP>& outputs, std::shared_ptr<Parameters> p = nullptr) override;

public:
    TrainInfo mTrainInfo;

private:
    Transformer::TrainConfig mConfig;
};

class InferOptimizer : public Express::Optimizer {
public:
    InferOptimizer(){}
    virtual Cost onMeasure(const std::vector<Express::VARP>& outputs, std::shared_ptr<Parameters> parameters = nullptr) override {
        Cost c;
        return c;
    }
    virtual bool onExecute(const std::vector<Express::VARP>& outputs, std::shared_ptr<Parameters> p = nullptr) override;
};
} // namespace Train
} // namespace MNN
#endif

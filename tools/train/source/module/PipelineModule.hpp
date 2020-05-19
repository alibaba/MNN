//
//  PipelineModule.hpp
//  MNN
//
//  Created by MNN on 2020/01/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef PipelineModule_hpp
#define PipelineModule_hpp
#include "Module.hpp"
#include "NN.hpp"
#include <MNN/expr/ExprCreator.hpp>
namespace MNN {
namespace Train {

class MNN_PUBLIC PipelineModule : public Module {
public:
    typedef std::function<std::pair<std::vector<int>, std::shared_ptr<Module>>(Express::EXPRP)> Transformer;
    static Module* extract(std::vector<Express::VARP> inputs, std::vector<Express::VARP> outputs, bool fortrain);
    static bool turnQuantize(Module* module, const int bits = 8, NN::FeatureScaleStatMethod featureScaleStatMethod = NN::PerTensor, NN::ScaleUpdateMethod scaleUpdateMethod = NN::MovingAverage);
    void toTrainQuant(const int bits = 8, NN::FeatureScaleStatMethod featureScaleStatMethod = NN::PerTensor,
                      NN::ScaleUpdateMethod scaleUpdateMethod = NN::MovingAverage);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    virtual void onClearCache() override;

private:
    PipelineModule(std::vector<Express::VARP> inputs, std::vector<Express::VARP> outputs,
                   const Transformer& transformFunction = {});
    std::vector<std::tuple<std::shared_ptr<Module>, std::vector<int>, std::vector<int>>> mSubModules;
    std::vector<Express::VARP> mStack;
    std::vector<int> mInputIndexes;
    std::vector<int> mOutputIndexes;
};
} // namespace Train
} // namespace MNN

#endif

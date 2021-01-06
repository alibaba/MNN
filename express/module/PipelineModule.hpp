//
//  PipelineModule.hpp
//  MNN
//
//  Created by MNN on 2020/01/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef PipelineModule_hpp
#define PipelineModule_hpp
#include <MNN/expr/Module.hpp>
#include <MNN/expr/NN.hpp>
#include <MNN/expr/ExprCreator.hpp>

namespace MNN {
struct Net;
}

namespace MNN {
namespace Express {

class MNN_PUBLIC PipelineModule : public Module {
public:
    typedef std::function<std::pair<std::vector<int>, std::shared_ptr<Module>>(Express::EXPRP)> Transformer;
    static Module* load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, const Module::Config* config = nullptr);
    static Module* extract(std::vector<Express::VARP> inputs, std::vector<Express::VARP> outputs, bool fortrain, const std::map<std::string, SubGraph>& subGraph = {});
    static bool turnQuantize(Module* module, const int bits = 8, NN::FeatureScaleStatMethod featureScaleStatMethod = NN::PerTensor, NN::ScaleUpdateMethod scaleUpdateMethod = NN::MovingAverage);
    void toTrainQuant(const int bits = 8, NN::FeatureScaleStatMethod featureScaleStatMethod = NN::PerTensor,
                      NN::ScaleUpdateMethod scaleUpdateMethod = NN::MovingAverage);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    virtual void onClearCache() override;
    std::vector<int> countOutputReference(std::vector<int> outputIndices);

private:
    PipelineModule(){}
    PipelineModule(std::vector<Express::VARP> inputs, std::vector<Express::VARP> outputs,
                   const Transformer& transformFunction = {});

    Module* clone(CloneContext* ctx) const override;

    std::vector<std::tuple<std::shared_ptr<Module>, std::vector<int>, std::vector<int>>> mSubModules;
    std::vector<int> mInputIndexes;
    std::vector<int> mOutputIndexes;
    int mStackSize = 0;
};
} // namespace Express
} // namespace MNN

#endif

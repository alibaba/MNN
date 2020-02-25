//
//  MergeOptimizer.hpp
//  MNN
//
//  Created by MNN on 2019/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef MergeOptimizer_hpp
#define MergeOptimizer_hpp

#include <MNN/expr/Optimizer.hpp>
#include <MNN/MNNForwardType.h>
namespace MNN {
namespace Express {
class MergeOptimizer : public Optimizer {
public:
    virtual ~MergeOptimizer() = default;
    MergeOptimizer(MNNForwardType type, int numberThread, BackendConfig* config);
    virtual Cost onMeasure(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters = nullptr)  override;
    
    //Modify the output directly, the parameters must be the same as onGetParameters
    virtual bool onExecute(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters = nullptr) override;

private:
    BackendConfig mConfig;
    MNNForwardType mType;
    int mNumberThread;
};
}; // namespace Express
}; // namespace MNN
#endif

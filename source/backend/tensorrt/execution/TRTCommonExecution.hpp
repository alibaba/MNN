//
//  TRTCommonExecution.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TRTCommonExecution_hpp
#define TRTCommonExecution_hpp
#include "TRTBackend.hpp"
#include "core/Execution.hpp"
#include "schema/current/MNNPlugin_generated.h"
using namespace std;
namespace MNN {

inline static std::shared_ptr<MNNTRTPlugin::PluginT> createPluginWithOutput(const std::vector<Tensor *> &outputs) {
    std::shared_ptr<MNNTRTPlugin::PluginT> plu(new MNNTRTPlugin::PluginT);
    plu->outputs.resize(outputs.size());
    for (int i = 0; i < outputs.size(); ++i) {
        auto shape = outputs[0]->shape();
        plu->outputs[i].reset(new MNNTRTPlugin::ShapeT);
        plu->outputs[i]->dim   = shape;
        plu->outputs[i]->bytes = outputs[i]->getType().bytes();
        plu->outputs[i]->type  = outputs[i]->getType().code;
    }
    return plu;
}

class TRTCommonExecution : public Execution {
public:
    TRTCommonExecution(Backend *backend, const Op *op);
    virtual ~TRTCommonExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    TRTBackend *mTrtBackend;
    const Op *mOp;
    std::vector<Tensor *> mInputs;
    std::vector<Tensor *> mOutputs;

    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) = 0;
};

} // namespace MNN
#endif /* TRTCommonExecution_hpp */

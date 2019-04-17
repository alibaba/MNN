//
//  MetalLSTM.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalLSTM_hpp
#define MetalLSTM_hpp

#import "Execution.hpp"
#import "MNN_generated.h"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalLSTM : public Execution {
public:
    MetalLSTM(Backend *backend, const LSTM *lstm);
    virtual ~MetalLSTM() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const LSTM *mLSTM;
    id<MTLBuffer> mWeightI;
    id<MTLBuffer> mWeightH;
    id<MTLBuffer> mBias;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalLSTM_hpp */

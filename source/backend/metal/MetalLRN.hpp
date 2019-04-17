//
//  MetalLRN.hpp
//  MNN
//
//  Created by MNN on 2019/01/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MetalLRN_hpp
#define MetalLRN_hpp

#import "Execution.hpp"
#import "MetalDefine.h"

#if MNN_METAL_ENABLED
namespace MNN {

class MetalLRN : public Execution {
public:
    MetalLRN(Backend *backend, int regionType, int localSize, float alpha, float beta);
    virtual ~MetalLRN() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mRegionType;
    int mLocalSize;
    float mAlpha;
    float mBeta;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
#endif /* MetalLRN_hpp */

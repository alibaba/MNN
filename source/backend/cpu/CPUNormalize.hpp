//
//  CPUNormalize.hpp
//  MNN
//
//  Created by MNN on 2018/07/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUNormalize_hpp
#define CPUNormalize_hpp

#include "core/AutoStorage.h"
#include "core/Execution.hpp"

namespace MNN {
class CPUNormalize : public Execution {
public:
    CPUNormalize(Backend *b, const MNN::Op *op);
    virtual ~CPUNormalize() = default;

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    Tensor mSummer;
    Tensor mSourceStorage;

    int32_t mAcrossSpatial;
    int32_t mChannelShared;
    float mEps;
    AutoStorage<float> mScale;
};
} // namespace MNN
#endif /* CPUNormalize_hpp */

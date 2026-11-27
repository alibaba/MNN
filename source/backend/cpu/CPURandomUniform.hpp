//
//  CPURandomUniform.h
//  MNN
//
//  Created by MNN on 2020/8/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPURandomUniform_h
#define CPURandomUniform_h

#include "core/Execution.hpp"

namespace MNN {
// RandomUniform and RandomNormal share the RandomUniform param table; low/high mean
// mean/scale for the normal variant. The values are copied out of the model because
// releaseModel() may free the model buffer while the session keeps running.
class CPURandomUniform : public Execution {
public:
    CPURandomUniform(Backend *b, const MNN::Op *op);
    virtual ~CPURandomUniform() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
private:
    float mLow;
    float mHigh;
    int mSeed;
    int mSeed2;
};

class CPURandomNormal : public Execution {
public:
    CPURandomNormal(Backend *b, const MNN::Op *op);
    virtual ~CPURandomNormal() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
private:
    float mLow;
    float mHigh;
    int mSeed;
    int mSeed2;
};

} // namespace MNN

#endif /* CPURandomUniform_h */

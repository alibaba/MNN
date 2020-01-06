//
//  CPUPool.hpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUPool_hpp
#define CPUPool_hpp

#include "backend/cpu/CPUBackend.hpp"

namespace MNN {
class CPUPool : public Execution {
public:
    CPUPool(Backend *b, const Pool *parameter);
    virtual ~CPUPool() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const Pool *mParameter;
    std::pair<int, std::function<void(int)> > mFunction;
};
    
class CPUPool3D : public Execution {
public:
    CPUPool3D(Backend *b, const Pool3D *param);
    virtual ~CPUPool3D() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::vector<int> mKernels;
    std::vector<int> mStrides;
    std::vector<int> mPads;
    PoolType mType;
    PoolPadType mPadType;
    std::shared_ptr<Tensor> mTempStorage;
};
} // namespace MNN

#endif /* CPUPool_hpp */

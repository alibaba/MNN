//
//  Arm82Moments.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#ifndef Arm82Moments_hpp
#define Arm82Moments_hpp

#include "Arm82Backend.hpp"
#include "core/Execution.hpp"

namespace MNN {

class Arm82Moments : public Execution {
public:
    Arm82Moments(Backend* backend, const MNN::Op* op);
    virtual ~Arm82Moments() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    void calculateMean(const FLOAT16 *src, FLOAT16 *mean, int channelBlock, int planeNumber);
    void calculateVariance(const FLOAT16 *src, const FLOAT16 *mean, FLOAT16* var, int channelBlock, int planeNumber);
    std::vector<int> mAxis;
    bool mKeepDims;
};

} // namespace MNN

#endif /* Arm82Moments_hpp */
#endif

//
//  CPULayerNorm.hpp
//  MNN
//
//  Created by MNN on 2023/07/11
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPULayerNorm_hpp
#define CPULayerNorm_hpp

#include "core/Execution.hpp"
#include "core/Macro.h"
namespace MNN {
class CPULayerNorm : public Execution {
public:
    explicit CPULayerNorm(const MNN::Op* op, Backend* backend);
    virtual ~CPULayerNorm();

    ErrorCode onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
    ErrorCode onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) override;
private:
    bool allocGammaBeta(int size);
private:
    int mAxis = 0;
    int mInnerSize = 1;
    int mOutterSize = 1;
    int mGroup = 1;
    float mEpsilon = 0.001;
    std::unique_ptr<Tensor> mGamma;
    std::unique_ptr<Tensor> mBeta;
    bool mIniGammaBeta = false;
    // LayerNormInt8 parameters.
    std::vector<float> mInpScale;
    std::vector<float> mOutScale;
    std::vector<ssize_t> mInpZero;
    std::vector<ssize_t> mOutZero;
    std::vector<ssize_t> mMaxMinValue;
};
} // namespace MNN
#endif /* CPULayerNorm_hpp */

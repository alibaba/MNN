//
//  CoreMLLayerNorm.hpp
//  MNN
//
//  Created by MNN on 2024/01/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLLAYERNORM_HPP
#define MNN_COREMLLAYERNORM_HPP

#include "CoreMLCommonExecution.hpp"
#include "CoreMLBackend.hpp"

namespace MNN {

class CoreMLLayerNorm : public CoreMLCommonExecution {
public:
    CoreMLLayerNorm(Backend *b, const Op *op, const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    virtual ~CoreMLLayerNorm() = default;
private:
    std::vector<int> mAxes;
    bool mRMS = false;
    float mEpsilon = 0.001;
    std::string mGamma, mBeta;
};
} // namespace MNN

#endif // MNN_COREMLLAYERNORM_HPP

//
//  CPUDetectionOutput.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDetectionOutput_hpp
#define CPUDetectionOutput_hpp

#include "core/Execution.hpp"

namespace MNN {

class CPUDetectionOutput : public Execution {
public:
    CPUDetectionOutput(Backend *backend, int classCount, float nmsThreshold, int keepTopK, float confidenceThreshold,
                       float objectnessScore);
    virtual ~CPUDetectionOutput() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    Tensor mLocation;
    Tensor mConfidence;
    Tensor mPriorbox;
    Tensor mArmLocation;
    Tensor mArmConfidence;
    int mClassCount;
    float mNMSThreshold;
    int mKeepTopK;
    float mConfidenceThreshold;
    float mObjectnessScoreThreshold;
};

} // namespace MNN

#endif /* CPUDetectionOutput_hpp */

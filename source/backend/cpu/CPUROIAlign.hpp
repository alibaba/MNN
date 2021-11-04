//
//  CPUROIAlign.hpp
//  MNN
//
//  Created by MNN on 2021/11/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUROIAlign_hpp
#define CPUROIAlign_hpp

#include "MNN_generated.h"
#include "core/Execution.hpp"

namespace MNN {
class CPUROIAlign : public Execution {
public:
    CPUROIAlign(Backend *backend, int pooledWidth, int pooledHeight, int samplingRatio, float spatialScale,
                bool aligned, PoolMode poolMode);
    virtual ~CPUROIAlign() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mPooledWidth;
    int mPooledHeight;
    int mSamplingRatio;
    float mSpatialScale;
    bool mAligned;
    PoolMode mPoolMode;

    Tensor mROI;
};

} // namespace MNN

#endif /* CPUROIAlign_hpp */
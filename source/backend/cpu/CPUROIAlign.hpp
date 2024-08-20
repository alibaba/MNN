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
                bool aligned, PoolType poolType, bool outputGrad);
    virtual ~CPUROIAlign() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    ErrorCode preCalcBilinearInterpolate(int height, int width, int pooledHeight, int pooledWidth, float roiStartH,
                                         float roiStartW, float binSizeH, float binSizeW, int samplingRatioH,
                                         int samplingRatioW, std::vector<std::vector<int>> &vecPos,
                                         std::vector<std::vector<float>> &vecArea);

protected:
    int mPooledWidth;
    int mPooledHeight;
    int mSamplingRatio;
    float mSpatialScale;
    bool mAligned;
    PoolType mPoolType;
    bool mOutputGrad; // false: output pooled value, true: output input grad

    Tensor mROI;
    std::shared_ptr<Tensor> mROITemp;
};

} // namespace MNN

#endif /* CPUROIAlign_hpp */

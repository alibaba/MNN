//
//  CPUROIPooling.hpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUROIPooling_hpp
#define CPUROIPooling_hpp

#include "core/Execution.hpp"

namespace MNN {

class CPUROIPooling : public Execution {
public:
    CPUROIPooling(Backend *backend, int pooledWidth, int pooledHeight, float spatialScale, bool outputGrad);
    virtual ~CPUROIPooling() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mPooledWidth;
    int mPooledHeight;
    float mSpatialScale;
    bool mOutputGrad; // false: output pooled value, true: output input grad

    Tensor mROI;
};

} // namespace MNN

#endif /* CPUROIPooling_hpp */

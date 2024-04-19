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
#include "CPUROIAlign.hpp"

namespace MNN {

class CPUROIPooling : public CPUROIAlign {
public:
    CPUROIPooling(Backend *backend, int pooledWidth, int pooledHeight, float spatialScale, bool outputGrad);
    virtual ~CPUROIPooling() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* CPUROIPooling_hpp */

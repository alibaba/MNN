//
//  CPUCropAndResize.hpp
//  MNN
//
//  Created by MNN on 2018/08/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUCropAndResize_hpp
#define CPUCropAndResize_hpp

#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
template <typename T>
class CPUCropAndResize : public Execution {
public:
    CPUCropAndResize(Backend* backend, const Op* op);
    ~CPUCropAndResize() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    const ErrorCode CropAndResize(const Tensor* image, const Tensor* boxes, const Tensor* boxIndex, Tensor* crops);
    CropAndResizeMethod mMethod;
    float mExtrapolationValue;
};

} // namespace MNN
#endif /* CPUCropAndResize_hpp */

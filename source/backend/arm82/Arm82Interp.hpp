//
//  Arm82Interp.hpp
//  MNN
//
//  Created by MNN on 2020/04/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#if defined(__ANDROID__) || defined(__aarch64__)

#ifndef CPUInterp_hpp
#define CPUInterp_hpp

#include "Arm82Backend.hpp"
#include "core/AutoStorage.h"
#include "core/Execution.hpp"

namespace MNN {
class Arm82Interp : public Execution {
public:
    Arm82Interp(Backend *backend, float widthScale, float heightScale, int resizeType, float widthOffset, float heightOffset);
    virtual ~Arm82Interp();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    Tensor mWidthPosition;
    Tensor mWidthFactor;
    Tensor mHeightPosition;
    Tensor mHeightFactor;
    Tensor mLineBuffer;
    float mWidthScale;
    float mHeightScale;
    float mWidthOffset;
    float mHeightOffset;
    int mResizeType;
    int mTheadNumbers;
};

} // namespace MNN

#endif
#endif

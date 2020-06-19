//
//  Arm82Interp.hpp
//  MNN
//
//  Created by MNN on 2020/04/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUInterp_hpp
#define CPUInterp_hpp

#include "backend/arm82/Arm82Backend.hpp"
#include "core/AutoStorage.h"
#include "core/Execution.hpp"

namespace MNN {
class Arm82Interp : public Execution {
public:
    Arm82Interp(Backend *backend, float widthScale, float heightScale, int resizeType, bool AlignCorners,
                bool halfPixelCenters);
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
    int mResizeType;
    int mTheadNumbers;
    bool mAlignCorners;
    bool mHalfPixelCenters;
};

} // namespace MNN

#endif

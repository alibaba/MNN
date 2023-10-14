//
//  CPUInterp.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUInterp_hpp
#define CPUInterp_hpp

#include "backend/cpu/CPUResize.hpp"

namespace MNN {

class CPUInterp : public CPUResizeCommon {
public:
    CPUInterp(Backend *backend, int resizeType,
              float widthScale = 0.f, float heightScale = 0.f,
              float widthOffset = 0.f, float heightOffset = 0.f);
    virtual ~CPUInterp();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

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
    int mResizeType; // 1:near 2: bilinear 3: cubic 4: nearest_round
    bool mInit = false;
    std::shared_ptr<Tensor> mInputTemp;
    std::shared_ptr<Tensor> mOutputTemp;
    int8_t mInputQuantZero = 0;
    int8_t mOutputQuantZero = 0;
    ssize_t mOutputQuantMIn = -127;
    ssize_t mOutputQuantMax = 127;
};

} // namespace MNN

#endif /* CPUInterp_hpp */

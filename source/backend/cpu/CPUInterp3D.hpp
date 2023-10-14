//
//  CPUInterp.hpp
//  MNN
//
//  Created by MNN on 2018/07/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUInterp3D_hpp
#define CPUInterp3D_hpp

#include "backend/cpu/CPUResize.hpp"

namespace MNN {

class CPUInterp3D : public CPUResizeCommon {
public:
    CPUInterp3D(Backend *backend, int resizeType,
                float widthScale = 0.f, float heightScale = 0.f, float depthScale = 0.f,
                float widthOffset = 0.f, float heightOffset = 0.f, float depthOffset = 0.f);
    virtual ~CPUInterp3D();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    Tensor mWidthPosition;
    Tensor mWidthFactor;
    Tensor mHeightPosition;
    Tensor mHeightFactor;
    Tensor mDepthPosition;
    Tensor mDepthFactor;
    Tensor mLineBuffer;
    float mWidthScale;
    float mHeightScale;
    float mDepthScale;
    float mWidthOffset;
    float mHeightOffset;
    float mDepthOffset;
    int mResizeType; // 1:near 2: bilinear 3: cubic 4: nearest_round
    bool mInit = false;
    std::shared_ptr<Tensor> mInputTemp;
    std::shared_ptr<Tensor> mOutputTemp;
};

} // namespace MNN

#endif /* CPUInterp_hpp */

//
//  GLConvolution.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDEMO_GLCONVOLUTION_H
#define MNNDEMO_GLCONVOLUTION_H

#include <functional>
#include "core/Execution.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLSSBOBuffer.hpp"
#include "backend/opengl/GLTexture.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace OpenGL {
class GPUConvolution : public Execution {
public:
    GPUConvolution(const Op *convOp, Backend *b);
    virtual ~GPUConvolution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    const Convolution2DCommon *mCommon;

    // In execute, use pad from mPadX and mPadY, don't use mCommon's pad
    mutable int mPadX;
    mutable int mPadY;

    int mInputDepth;
};

class GLConvolution : public GPUConvolution {
public:
    GLConvolution(const std::vector<Tensor *> &inputs, const Op *convOp, Backend *b);
    virtual ~GLConvolution();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLTexture> mKernelTexture;
    std::shared_ptr<GLSSBOBuffer> mBiasBuffer;
    std::shared_ptr<GLProgram> mProgram;
    bool mIs1x1 = false;
    int mLocalSize[3];
    GLBackend* mBackend;
    int mKx, mKy, mSx, mSy, mDx, mDy;
};
} // namespace OpenGL
} // namespace MNN

#endif // MNNDEMO_GLCONVOLUTION_H

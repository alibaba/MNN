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
#include "Execution.hpp"
#include "GLProgram.h"
#include "GLSSBOBuffer.h"
#include "GLTexture.h"
#include "MNN_generated.h"
namespace MNN {
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

    int mSrcCount;
};
class GLConvolution : public GPUConvolution {
public:
    GLConvolution(const Op *convOp, Backend *b);
    virtual ~GLConvolution();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLTexture> mKernelTexture;
    std::shared_ptr<GLSSBOBuffer> mBiasBuffer;
    std::shared_ptr<GLProgram> mProgram;
    bool mIs1x1 = false;
    int mLocalSize[3];
    std::function<void()> mSetUniform;
};
} // namespace MNN

#endif // MNNDEMO_GLCONVOLUTION_H

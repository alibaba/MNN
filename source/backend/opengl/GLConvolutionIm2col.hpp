//
//  GLConvolutionIm2col.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLCONVOLUTION_IM2COL_H
#define GLCONVOLUTION_IM2COL_H

#include <functional>
#include "core/Execution.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLSSBOBuffer.hpp"
#include "backend/opengl/GLTexture.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace OpenGL {

class GLConvolutionIm2col : public GPUConvolution {
public:
    GLConvolutionIm2col(const std::vector<Tensor *> &inputs, const Op *convOp, Backend *b);
    virtual ~GLConvolutionIm2col();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    GLBackend * mGLBackend;
    std::shared_ptr<GLTexture> mKernelTexture;
    std::shared_ptr<GLTexture> mSrcTexture;
    std::shared_ptr<GLTexture> mDstTexture;
    std::shared_ptr<GLSSBOBuffer> mBiasBuffer;
    std::shared_ptr<GLProgram> mIm2ColProgram;
    std::shared_ptr<GLProgram> mGemm16x16Program;
    std::shared_ptr<GLProgram> mCol2ImProgram;
    std::function<void()> mImage2ColUniform;
    int obxohxow_4;
    int mIm2colSize[3];
    int mGemmSize[3];
    int mCol2imSize[3];
    bool mIsConv1x1;
};
} // namespace OpenGL
} // namespace MNN

#endif // GLCONVOLUTION_IM2COL_H

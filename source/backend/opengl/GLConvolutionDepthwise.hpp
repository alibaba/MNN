//
//  GLConvolutionDepthwise.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDEMO_GLCONVOLUTIONDEPTHWISE_H
#define MNNDEMO_GLCONVOLUTIONDEPTHWISE_H

#include "core/Execution.hpp"
#include "backend/opengl/GLConvolution.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace OpenGL {
class GLConvolutionDepthwise : public GPUConvolution {
public:
    GLConvolutionDepthwise(const std::vector<Tensor *> &inputs, const Op *op, Backend *b);
    virtual ~GLConvolutionDepthwise();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLTexture> mKernelTexture;
    std::shared_ptr<GLSSBOBuffer> mBiasBuffer;
    std::shared_ptr<GLProgram> mProgram;
    std::function<void()> mSetUniform;
};

} // namespace OpenGL
} // namespace MNN

#endif // MNNDEMO_GLCONVOLUTION_H

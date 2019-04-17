//
//  GLConvolutionDepthwise.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDEMO_GLCONVOLUTIONDEPTHWISE_H
#define MNNDEMO_GLCONVOLUTIONDEPTHWISE_H

#include "Execution.hpp"
#include "GLConvolution.h"
#include "MNN_generated.h"

namespace MNN {

class GLConvolutionDepthwise : public GPUConvolution {
public:
    GLConvolutionDepthwise(const Op *convOp, Backend *b);
    virtual ~GLConvolutionDepthwise();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLTexture> mKernelTexture;
    std::shared_ptr<GLSSBOBuffer> mBiasBuffer;
    std::shared_ptr<GLProgram> mProgram;
    std::function<void()> mSetUniform;
};

} // namespace MNN

#endif // MNNDEMO_GLCONVOLUTION_H

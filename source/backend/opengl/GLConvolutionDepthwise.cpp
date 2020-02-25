//
//  GLConvolutionDepthwise.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLConvolutionDepthwise.hpp"
#include <MNN/AutoTime.hpp>

#include <sstream>
#include "AllShader.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "core/Macro.h"
namespace MNN {
namespace OpenGL {

static const int gXLocal = 8;
static const int gYLocal = 8;
static const int gZLocal = 1;

GLConvolutionDepthwise::~GLConvolutionDepthwise() {
}

GLConvolutionDepthwise::GLConvolutionDepthwise(const std::vector<Tensor *> &inputs, const Op *convOp, Backend *bn) : GPUConvolution(convOp, bn) {
    auto extra = (GLBackend *)bn;

    mBiasBuffer.reset(new GLSSBOBuffer(sizeof(float) * ALIGN_UP4(mCommon->outputCount())));
    int fw           = mCommon->kernelX();
    int fh           = mCommon->kernelY();
    int unit         = 4;
    int srcDepthQuad = UP_DIV(mInputDepth, unit);

    auto kernelBuffer = std::shared_ptr<GLSSBOBuffer>(new GLSSBOBuffer(sizeof(float) * fw * fh * srcDepthQuad * 4));
    auto weight       = kernelBuffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    if(weight != nullptr){
        ::memset(weight, 0, fw * fh * srcDepthQuad * 4 * sizeof(float));
        ::memcpy(weight, convOp->main_as_Convolution2D()->weight()->data(),
                 convOp->main_as_Convolution2D()->weight()->size() * sizeof(float));
    }

    kernelBuffer->unmap();

    auto bias = mBiasBuffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    if(bias != nullptr){
        ::memset(bias, 0, ALIGN_UP4(mCommon->outputCount()) * sizeof(float));
        ::memcpy(bias, convOp->main_as_Convolution2D()->bias()->data(),
                 convOp->main_as_Convolution2D()->bias()->size() * sizeof(float));
    }
    mBiasBuffer->unmap();

    std::vector<std::string> prefix;
    if (mCommon->relu()) {
        prefix.push_back("#define RELU");
    }
    if (mCommon->relu6()) {
        prefix.push_back("#define RELU6");
    }

    {
        std::ostringstream os;
        os << "#define XLOCAL " << gXLocal;
        prefix.push_back(os.str());
    }
    {
        std::ostringstream os;
        os << "#define YLOCAL " << gYLocal;
        prefix.push_back(os.str());
    }
    {
        std::ostringstream os;
        os << "#define ZLOCAL " << gZLocal;
        prefix.push_back(os.str());
    }

    mProgram       = extra->getProgram("convolution_depthwise", glsl_convlutionDepthwise_glsl, prefix);
    mKernelTexture = std::shared_ptr<GLTexture>(new GLTexture(srcDepthQuad, fw, fh, ((GLBackend *)backend())->getTextrueFormat(), GL_TEXTURE_3D, false));

    auto transform = extra->getProgram("transform_kernel_image_depthwise", glsl_kernel2ImageDepthwise_glsl);
    transform->useProgram();
    glBindImageTexture(0, mKernelTexture->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, kernelBuffer->getId());
    OPENGL_CHECK_ERROR;
    glUniform1i(3, fw);
    glUniform1i(4, fh);
    OPENGL_CHECK_ERROR;

    ((GLBackend *)backend())->compute(srcDepthQuad, fw, fh);
    OPENGL_CHECK_ERROR;

}

ErrorCode GLConvolutionDepthwise::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    GPUConvolution::onResize(inputs, outputs);
    int kx      = mCommon->kernelX();
    int ky      = mCommon->kernelY();
    int sx      = mCommon->strideX();
    int sy      = mCommon->strideY();
    int dx      = mCommon->dilateX();
    int dy      = mCommon->dilateY();
    mSetUniform = [=]() {
        glUniform2i(4, mPadX, mPadY);
        glUniform2i(5, kx, ky);
        glUniform2i(6, sx, sy);
        glUniform2i(7, dx, dy);
    };
    return NO_ERROR;
}

ErrorCode GLConvolutionDepthwise::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    {
        auto convLayer = mCommon;

        auto input         = inputs[0];
        auto output        = outputs[0];
        auto inputTexture  = input->deviceId();
        auto outputTexture = output->deviceId();
        int dst_depth_quad = UP_DIV(output->channel(), 4);

        mProgram->useProgram();
        glBindImageTexture(0, outputTexture, 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
        OPENGL_CHECK_ERROR;
        {
            int texId = 0;
            glActiveTexture(GL_TEXTURE0 + texId);
            glUniform1i(1, texId);
            glBindTexture(GL_TEXTURE_3D, inputTexture);
            OPENGL_CHECK_ERROR;
        }
        {
            int texId = 1;
            glActiveTexture(GL_TEXTURE0 + texId);
            OPENGL_CHECK_ERROR;
            glUniform1i(2, texId);

            OPENGL_CHECK_ERROR;
            glBindTexture(GL_TEXTURE_3D, mKernelTexture->id());
            OPENGL_CHECK_ERROR;
        }
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mBiasBuffer->getId());

        OPENGL_CHECK_ERROR;
        mSetUniform();

        glUniform3i(10, output->width(), output->height(), UP_DIV(output->channel(), 4));
        glUniform3i(11, input->width(), input->height(), UP_DIV(input->channel(), 4));

        OPENGL_CHECK_ERROR;

        ((GLBackend *)backend())->compute(UP_DIV(output->width(), (gXLocal)), UP_DIV(output->height(), gYLocal),
                          UP_DIV(dst_depth_quad, gZLocal));
        OPENGL_CHECK_ERROR;

    }

    return NO_ERROR;
}
GLCreatorRegister<TypedCreator<GLConvolutionDepthwise>> __depthwise_conv_op(OpType_ConvolutionDepthwise);
} // namespace OpenGL
} // namespace MNN

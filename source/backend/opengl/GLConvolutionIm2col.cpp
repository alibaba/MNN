//
//  GLConvolution.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLConvolution.hpp"
#include <MNN/AutoTime.hpp>

#include <sstream>
#include "AllShader.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "core/Macro.h"
#include "backend/opengl/GLConvolutionIm2col.hpp"
#include "backend/opengl/GLUtils.hpp"
namespace MNN {
namespace OpenGL {

GLConvolutionIm2col::~GLConvolutionIm2col() {
}

#define UNIT 4
#define UNIT2 16
GLConvolutionIm2col::GLConvolutionIm2col(const std::vector<Tensor *> &inputs, const Op *convOp, Backend *bn) : GPUConvolution(convOp, bn) {
    auto totalWeightSize = ALIGN_UP4(mCommon->outputCount()) * ALIGN_UP4(mInputDepth) * (mCommon->kernelY() * mCommon->kernelX());
    mGLBackend = (GLBackend *)bn;
    auto mKernelBuffer = std::shared_ptr<GLSSBOBuffer>(new GLSSBOBuffer(sizeof(float) * totalWeightSize));
    int fw                = mCommon->kernelX();
    int fh                = mCommon->kernelY();
    mIsConv1x1 = (fw == 1 && fh == 1) ? true : false;
    int oc_4         = UP_DIV(mCommon->outputCount(), UNIT);
    int ic_4      = UP_DIV(mInputDepth, UNIT);
    float *dest           = (float *)mKernelBuffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    if(NULL != dest){
        ::memset(dest, 0, totalWeightSize * sizeof(float));
        const float *source = convOp->main_as_Convolution2D()->weight()->data();
        int cur             = 0;

        //weight : oc ic -> oc/4 ic/4 ic4 oc4
        //weight image : oc_4, ic_4 * ic4 oc4
        int alignedWeightSize = ic_4 * fw * fh * UNIT2;
        for (int b = 0; b < mCommon->outputCount(); ++b) {
            int b_4      = b / UNIT;
            float *dst_b = dest + b_4 * alignedWeightSize;
            int mx       = b % UNIT;
            for (int d = 0; d < mInputDepth; ++d) {
                int my       = d % UNIT;
                int d_4      = d / UNIT;
                float *dst_d = dst_b + d_4 * fw * fh * UNIT2;
                for (int y = 0; y < fh; ++y) {
                    float *dst_y = dst_d + y * fw * UNIT2;
                    for (int x = 0; x < fw; ++x) {
                        float *dst_x          = dst_y + x * UNIT2;
                        dst_x[UNIT * my + mx] = source[cur++];
                    }
                }
            }
        }
    }else{
        MNN_ASSERT(NULL != dest);
    }

    mKernelBuffer->unmap();

    mKernelTexture = std::shared_ptr<GLTexture>(new GLTexture(ic_4 * UNIT*fw*fh, oc_4, 1, ((GLBackend *)backend())->getTextrueFormat(), GL_TEXTURE_2D, false));
    auto transform = mGLBackend->getProgram("transform_kernel_image", glsl_kernel2image_glsl);
    int imageWidth = ROUND_UP(mInputDepth, 4)*fw*fh;
    int imageHeight = oc_4;
    transform->useProgram();
    glBindImageTexture(0, mKernelTexture->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mKernelBuffer->getId());
    OPENGL_CHECK_ERROR;
    glUniform1i(3, imageWidth);
    OPENGL_CHECK_ERROR;
    glUniform1i(4, imageHeight);
    OPENGL_CHECK_ERROR;
    ((GLBackend *)backend())->compute(UP_DIV(imageWidth, 4), UP_DIV(oc_4, 4), 1);
    OPENGL_CHECK_ERROR;

//bias
    mBiasBuffer.reset(new GLSSBOBuffer(sizeof(float) * ALIGN_UP4(mCommon->outputCount())));
    float* bias = (float*)(mBiasBuffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
    if(bias != nullptr){
        ::memset(bias, 0, ALIGN_UP4(mCommon->outputCount()) * sizeof(float));
        ::memcpy(bias, convOp->main_as_Convolution2D()->bias()->data(),
                 convOp->main_as_Convolution2D()->bias()->size() * sizeof(float));
    }
    mBiasBuffer->unmap();
}

ErrorCode GLConvolutionIm2col::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    GPUConvolution::onResize(inputs, outputs);
    std::vector<std::string> im2colPrefix;
    std::vector<std::string> gemmPrefix;
    std::vector<std::string> col2imPrefix;

    if (mCommon->relu()) {
        im2colPrefix.push_back("#define RELU");
        gemmPrefix.push_back("#define RELU");
        col2imPrefix.push_back("#define RELU");
    }
    if (mCommon->relu6()) {
        im2colPrefix.push_back("#define RELU6");
        gemmPrefix.push_back("#define RELU6");
        col2imPrefix.push_back("#define RELU6");
    }

    int ob = outputs[0]->batch();
    int oc = outputs[0]->channel();
    int oh = outputs[0]->height();
    int ow = outputs[0]->width();

    int ic = inputs[0]->channel();

    obxohxow_4  = UP_DIV(ob*oh*ow, 4);

    int fw                = mCommon->kernelX();
    int fh                = mCommon->kernelY();

    //input : temp image : (ib*oh*ow)/ 4, ic/4*(ib*oh*ow)%4*ic4
    //output : temp image : oc/4 * (ob*oh*ow)%4, (ob*oh*ow)/4 * oc4
    mSrcTexture = std::shared_ptr<GLTexture>(new GLTexture(UP_DIV(ic, 4)*UNIT*fw*fh, obxohxow_4, 1, ((GLBackend *)backend())->getTextrueFormat(), GL_TEXTURE_2D, false));
    mDstTexture = std::shared_ptr<GLTexture>(new GLTexture(obxohxow_4, UP_DIV(oc, 4) * UNIT, 1, ((GLBackend *)backend())->getTextrueFormat(), GL_TEXTURE_2D, false));

    auto transform = mGLBackend->getProgram("clear_texture", glsl_clear_texture_glsl);
    transform->useProgram();
    glBindImageTexture(0, mSrcTexture->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
    OPENGL_CHECK_ERROR;
    glUniform1i(1, UP_DIV(ic, 4)*UNIT*fw*fh);
    OPENGL_CHECK_ERROR;
    glUniform1i(2, obxohxow_4);
    OPENGL_CHECK_ERROR;
    ((GLBackend *)backend())->compute(UP_DIV(UP_DIV(ic, 4)*UNIT*fw*fh, 4), UP_DIV(obxohxow_4, 4), 1);
    OPENGL_CHECK_ERROR;

    if (true == mIsConv1x1) {
        setLocalSize(im2colPrefix, mIm2colSize, 8, 8, 1);
        mIm2ColProgram = mGLBackend->getProgram("image2col1x1", glsl_im2col1x1_glsl, im2colPrefix);
    }else{
        setLocalSize(im2colPrefix, mIm2colSize, 8, 8, 1);
        mIm2ColProgram = mGLBackend->getProgram("image2col", glsl_im2col_glsl, im2colPrefix);
    }

    setLocalSize(gemmPrefix, mGemmSize, 8, 8, 1);
    mGemm16x16Program = mGLBackend->getProgram("gemm16x16", glsl_gemm16x16_glsl, gemmPrefix);
    setLocalSize(col2imPrefix, mCol2imSize, 8, 8, 1);
    mCol2ImProgram = mGLBackend->getProgram("col2image", glsl_col2im_glsl, col2imPrefix);
    if (!mIsConv1x1) {
        mImage2ColUniform = [=]() {
            glUniform2i(2, mPadX, mPadY);
            glUniform2i(3, mCommon->kernelX(), mCommon->kernelY());
            glUniform2i(4, mCommon->strideX(), mCommon->strideY());
            glUniform2i(5, mCommon->dilateX(), mCommon->dilateY());
        };
    }

    return NO_ERROR;
}
ErrorCode GLConvolutionIm2col::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    auto input         = inputs[0];
    auto output        = outputs[0];
    auto inputTexture  = input->deviceId();
    auto outputTexture = output->deviceId();

    int iw = input->width();
    int ih = input->height();
    int ic = input->channel();
    int ib = input->batch();

    int ow = output->width();
    int oh = output->height();
    int oc = output->channel();
    int ob = output->batch();

    int ic_4 = UP_DIV(ic, 4);
    int oc_4 = UP_DIV(oc, 4);

    //        image2col
    {
        mIm2ColProgram->useProgram();
        glBindImageTexture(0, mSrcTexture->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
        {
            int texId = 0;
            glActiveTexture(GL_TEXTURE0 + texId);
            glUniform1i(1, texId);
            glBindTexture(GL_TEXTURE_3D, inputTexture);
            OPENGL_CHECK_ERROR;
        }

        if (mIsConv1x1) {
            glUniform1i(5, ic_4);
            glUniform1i(6, ow);
            glUniform1i(7, oh);
        }else{
            mImage2ColUniform();
            glUniform4i(6, iw, ih, ic_4, 1);
            glUniform4i(7, ow, oh, oc_4, 1);
        }
        OPENGL_CHECK_ERROR;
        ((GLBackend *)backend())->compute(UP_DIV(ow, mIm2colSize[0]), UP_DIV(oh, mIm2colSize[1]), UP_DIV(ic_4*ib, mIm2colSize[2]));
        OPENGL_CHECK_ERROR;
    }

    //gemm
    {
        mGemm16x16Program->useProgram();
        glBindImageTexture(0, mDstTexture->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
        OPENGL_CHECK_ERROR;
        glBindImageTexture(1, mSrcTexture->id(), 0, GL_TRUE, 0, GL_READ_ONLY, ((GLBackend *)backend())->getTextrueFormat());
        glBindImageTexture(2, mKernelTexture->id(), 0, GL_TRUE, 0, GL_READ_ONLY, ((GLBackend *)backend())->getTextrueFormat());
        glUniform2i(3, obxohxow_4, oc_4);
        if (mIsConv1x1) {
            glUniform1i(4, ic_4);
        }else{
            glUniform1i(4, ic_4*mCommon->kernelX()*mCommon->kernelY());
        }
        OPENGL_CHECK_ERROR;
        ((GLBackend *)backend())->compute(UP_DIV(obxohxow_4, mGemmSize[0]), UP_DIV(oc_4, mGemmSize[1]), 1);
        OPENGL_CHECK_ERROR;
    }

    //col2image
    {
        mCol2ImProgram->useProgram();
        OPENGL_CHECK_ERROR;
        glBindImageTexture(0, outputTexture, 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
        {
            int texId = 0;
            glActiveTexture(GL_TEXTURE0 + texId);
            glUniform1i(1, texId);
            glBindTexture(GL_TEXTURE_2D, mDstTexture->id());
            OPENGL_CHECK_ERROR;
        }
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mBiasBuffer->getId());
        OPENGL_CHECK_ERROR;
        glUniform3i(3, ow, oh, oc_4);
        OPENGL_CHECK_ERROR;
        ((GLBackend *)backend())->compute(UP_DIV(ow, mCol2imSize[0]), UP_DIV(oh, mCol2imSize[1]), UP_DIV(oc_4*ob, mCol2imSize[2]));
        OPENGL_CHECK_ERROR;
    }

    return NO_ERROR;
}

} // namespace OpenGL
} // namespace MNN

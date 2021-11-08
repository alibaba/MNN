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
#include "core/Macro.h"
#include "backend/opengl/GLConvolutionIm2col.hpp"
namespace MNN {
namespace OpenGL {

#define UNIT 4

GPUConvolution::GPUConvolution(const Op *convOp, Backend *b) : MNN::Execution(b) {
    mCommon          = convOp->main_as_Convolution2D()->common();
    auto convReal    = convOp->main_as_Convolution2D();
    auto outputCount = mCommon->outputCount();
    mInputDepth        = 0;

    if (convReal->weight() != NULL) {
        auto weightSize = convReal->weight()->size();
        mInputDepth       = weightSize * mCommon->group() / mCommon->kernelX() / mCommon->kernelY() / outputCount;
    }
}
GPUConvolution::~GPUConvolution() {
}

ErrorCode GPUConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    if (mCommon->padMode() == PadMode_SAME) {
        int kernelWidthSize = (mCommon->kernelX() - 1) * mCommon->dilateX() + 1;
        int kernelHeightSize = (mCommon->kernelY() - 1) * mCommon->dilateY() + 1;
        int pad_needed_width  = (output->width() - 1) * mCommon->strideX() + kernelWidthSize - input->width();
        int pad_needed_height = (output->height() - 1) * mCommon->strideY() + kernelHeightSize - input->height();

        mPadX = (pad_needed_width > 0 ?  pad_needed_width : 0) / 2;
        mPadY = (pad_needed_height > 0 ?  pad_needed_height : 0) / 2;
        return NO_ERROR;
    }
    mPadX = mCommon->padX();
    mPadY = mCommon->padY();

    return NO_ERROR;
}

GLConvolution::~GLConvolution() {
}

GLConvolution::GLConvolution(const std::vector<Tensor *> &inputs, const Op *convOp, Backend *bn) : GPUConvolution(convOp, bn) {
    auto totalWeightSize =
        ALIGN_UP4(mCommon->outputCount()) * ALIGN_UP4(mInputDepth) * (mCommon->kernelY() * mCommon->kernelX());
    auto extra = (GLBackend *)bn;

    mBiasBuffer.reset(new GLSSBOBuffer(sizeof(float) * ALIGN_UP4(mCommon->outputCount())));
    float* bias = (float*)(mBiasBuffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
    if(bias != nullptr){
        ::memset(bias, 0, ALIGN_UP4(mCommon->outputCount()) * sizeof(float));
        ::memcpy(bias, convOp->main_as_Convolution2D()->bias()->data(),
                 convOp->main_as_Convolution2D()->bias()->size() * sizeof(float));
    }
    mBiasBuffer->unmap();

    auto mKernelBuffer = std::shared_ptr<GLSSBOBuffer>(new GLSSBOBuffer(sizeof(float) * totalWeightSize));
    int fw                = mCommon->kernelX();
    int fh                = mCommon->kernelY();
    int unit              = 4;
    int unit2             = unit * unit;
    int alignedWeightSize = UP_DIV(mInputDepth, unit) * fw * fh * unit2;
    int oc_4         = UP_DIV(mCommon->outputCount(), unit);

    float *dest           = (float *)mKernelBuffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    if(dest != nullptr){
        ::memset(dest, 0, alignedWeightSize * sizeof(float));
        const float *source = convOp->main_as_Convolution2D()->weight()->data();
        int cur             = 0;

        //weight : oc ic h w -> oc/4, ic/4 ky kx ic4 oc4
        for (int b = 0; b < mCommon->outputCount(); ++b) {
            int b_4      = b / unit;
            float *dst_b = dest + b_4 * alignedWeightSize;
            int mx       = b % unit;
            for (int d = 0; d < mInputDepth; ++d) {
                int my       = d % unit;
                int d_4      = d / unit;
                float *dst_d = dst_b + d_4 * fw * fh * unit2;
                for (int y = 0; y < fh; ++y) {
                    float *dst_y = dst_d + y * fw * unit2;
                    for (int x = 0; x < fw; ++x) {
                        float *dst_x          = dst_y + x * unit2;
                        dst_x[unit * my + mx] = source[cur++];
                    }
                }
            }
        }
    }

    mKernelBuffer->unmap();

    int ic_4      = UP_DIV(mInputDepth, unit);
    //weight image : ky kx, oc/4, ic/4*ic4 oc4
    mKernelTexture =
    std::shared_ptr<GLTexture>(new GLTexture(ic_4 * unit, oc_4, fw * fh, ((GLBackend *)backend())->getTextrueFormat() , GL_TEXTURE_3D, false));

    auto transform = extra->getProgram("transform_kernel_image_adreno", glsl_kernel2image_adreno_glsl);
    transform->useProgram();
    glBindImageTexture(0, mKernelTexture->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mKernelBuffer->getId());
    OPENGL_CHECK_ERROR;
    glUniform1i(3, fw * fh);
    glUniform1i(4, ic_4);
    OPENGL_CHECK_ERROR;

    ((GLBackend *)backend())->compute(ic_4, oc_4, fw * fh);
    OPENGL_CHECK_ERROR;
}

ErrorCode GLConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    GPUConvolution::onResize(inputs, outputs);
    auto extra = (GLBackend *)backend();
    std::vector<std::string> prefix;
    if (mCommon->relu()) {
        prefix.push_back("#define RELU");
    }
    if (mCommon->relu6()) {
        prefix.push_back("#define RELU6");
    }

    auto dstDepthQuad = UP_DIV(outputs[0]->channel(), 4);

    setLocalSize(prefix, mLocalSize, 1, 1, dstDepthQuad);

    if (1 == mCommon->kernelY() && 1 == mCommon->kernelX() && 1 == mCommon->strideY() && 1 == mCommon->strideX() &&
        0 == mCommon->padX() && 0 == mCommon->padY()) {
        mIs1x1      = true;
    }

    if (mIs1x1) {
        mProgram = extra->getProgram("convolution1x1", glsl_convolution1x1_glsl, prefix);
    } else {
        mKx      = mCommon->kernelX();
        mKy      = mCommon->kernelY();
        mSx      = mCommon->strideX();
        mSy      = mCommon->strideY();
        mDx      = mCommon->dilateX();
        mDy      = mCommon->dilateY();
        mProgram = extra->getProgram("convolution", glsl_convolution_glsl, prefix);
    }

    return NO_ERROR;
}


ErrorCode GLConvolution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    {
        auto convLayer = mCommon;

        auto input         = inputs[0];
        auto output        = outputs[0];
        auto inputTexture  = input->deviceId();
        auto outputTexture = output->deviceId();
        int oc_4 = UP_DIV(output->channel(), 4);

        mProgram->useProgram();
        glBindImageTexture(0, outputTexture, 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
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

        if(!mIs1x1){
            glUniform2i(4, mPadX, mPadY);
            glUniform2i(5, mKx, mKy);
            glUniform2i(6, mSx, mSy);
            glUniform2i(7, mDx, mDy);
        }
        OPENGL_CHECK_ERROR;
        glUniform3i(10, output->width(), output->height(), UP_DIV(output->channel(), 4));
        glUniform3i(11, input->width(), input->height(), UP_DIV(input->channel(), 4));

        glUniform1i(8, UNIT);
        OPENGL_CHECK_ERROR;

        ((GLBackend *)backend())->compute(UP_DIV(output->width(), UNIT*mLocalSize[0]), UP_DIV(output->height(), mLocalSize[1]),
                                                UP_DIV(oc_4, mLocalSize[2]));

        OPENGL_CHECK_ERROR;
    }

    return NO_ERROR;
}


class ConvolutionCreator : public GLBackend::Creator {
public:
    virtual ~ConvolutionCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto common = op->main_as_Convolution2D()->common();

        //TODO: bugfix
        if(common->padX() == 1 || common->strideX() != 1){
            return new GLConvolution(inputs, op, backend);
        }
        if(((GLBackend *)backend)->gpuType() == GLBackend::ADRENO){
            if(((GLBackend *)backend)->glVersion() >= 269){
                return new GLConvolution(inputs, op, backend);
            }else{
                return new GLConvolutionIm2col(inputs, op, backend);
            }
        }else{
            return new GLConvolutionIm2col(inputs, op, backend);
        }
    }
};

GLCreatorRegister<ConvolutionCreator> __gl_conv_op(OpType_Convolution);
} // namespace OpenGL
} // namespace MNN

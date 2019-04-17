//
//  GLConvolution.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GLConvolution.h"
#include "AutoTime.hpp"

#include <sstream>
#include "AllShader.h"
#include "GLBackend.h"
#include "Macro.h"
namespace MNN {
static const int gD1Unroll = 4;

GPUConvolution::GPUConvolution(const Op *convOp, Backend *b) : MNN::Execution(b) {
    mCommon          = convOp->main_as_Convolution2D()->common();
    auto convReal    = convOp->main_as_Convolution2D();
    auto outputCount = mCommon->outputCount();
    mSrcCount        = 0;

    if (convReal->weight() != NULL) {
        auto weightSize = convReal->weight()->size();
        mSrcCount       = weightSize * mCommon->group() / mCommon->kernelX() / mCommon->kernelY() / outputCount;
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

        mPadX = pad_needed_width / 2;
        mPadY = pad_needed_height / 2;
        return NO_ERROR;
    }
    mPadX = mCommon->padX();
    mPadY = mCommon->padY();

    return NO_ERROR;
}

GLConvolution::~GLConvolution() {
}

GLConvolution::GLConvolution(const Op *convOp, Backend *bn) : GPUConvolution(convOp, bn) {
    auto srcCount = mSrcCount;
    auto totalWeightSize =
        ALIGN_UP4(mCommon->outputCount()) * ALIGN_UP4(srcCount) * (mCommon->kernelY() * mCommon->kernelX());
    auto extra = (GLBackend *)bn;

    mBiasBuffer.reset(new GLSSBOBuffer(sizeof(float) * ALIGN_UP4(mCommon->outputCount())));
    auto mKernelBuffer = std::shared_ptr<GLSSBOBuffer>(new GLSSBOBuffer(sizeof(float) * totalWeightSize));

    auto bias = mBiasBuffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    ::memset(bias, 0, ALIGN_UP4(mCommon->outputCount()) * sizeof(float));
    ::memcpy(bias, convOp->main_as_Convolution2D()->bias()->data(),
             convOp->main_as_Convolution2D()->bias()->size() * sizeof(float));
    mBiasBuffer->unmap();
    auto weight           = mKernelBuffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
    auto layer            = mCommon;
    int fw                = layer->kernelX();
    int fh                = layer->kernelY();
    int unit              = 4;
    int unit2             = 4 * 4;
    int depth             = mSrcCount;
    int srcDepthQuad      = UP_DIV(mSrcCount, unit);
    int alignedWeightSize = UP_DIV(layer->outputCount(), unit) * UP_DIV(depth, unit) * fw * fh * unit2;
    float *dest           = (float *)weight;
    MNN_ASSERT(NULL != dest);
    const float *source = convOp->main_as_Convolution2D()->weight()->data();
    int cur             = 0;
    int batch_4         = ALIGN_UP4(layer->outputCount()) / unit;
    for (int b = 0; b < layer->outputCount(); ++b) {
        int b_4      = b / unit;
        float *dst_b = dest + b_4 * (alignedWeightSize / batch_4);
        int mx       = b % unit;
        for (int d = 0; d < depth; ++d) {
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
    mKernelBuffer->unmap();
    mKernelTexture =
        std::shared_ptr<GLTexture>(new GLTexture(4 * srcDepthQuad, batch_4, fw * fh, GL_TEXTURE_3D, false));

    auto transform = extra->getProgram("transform_kernel_image", glsl_kernel2image_glsl);
    transform->use();
    glBindImageTexture(0, mKernelTexture->id(), 0, GL_TRUE, 0, GL_WRITE_ONLY, TEXTURE_FORMAT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mKernelBuffer->getId());
    OPENGL_CHECK_ERROR;
    glUniform1i(3, fw * fh);
    glUniform1i(4, srcDepthQuad);
    OPENGL_CHECK_ERROR;

    glDispatchCompute(srcDepthQuad, batch_4, fw * fh);
    OPENGL_CHECK_ERROR;
}

ErrorCode GLConvolution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    //    glFinish();
    //    for (int i = 0; i < 10; ++i)
    {
        AUTOTIME;
        auto convLayer = mCommon;

        auto input         = inputs[0];
        auto output        = outputs[0];
        auto inputTexture  = input->deviceId();
        auto outputTexture = output->deviceId();
        int dst_depth_quad = UP_DIV(output->channel(), 4);

        mProgram->use();
        glBindImageTexture(0, outputTexture, 0, GL_TRUE, 0, GL_WRITE_ONLY, TEXTURE_FORMAT);
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

        glUniform1i(8, gD1Unroll);
        OPENGL_CHECK_ERROR;

        glDispatchCompute(UP_DIV(output->width(), (gD1Unroll * mLocalSize[0])), UP_DIV(output->height(), mLocalSize[1]),
                          UP_DIV(dst_depth_quad, mLocalSize[2]));
        OPENGL_CHECK_ERROR;
#ifdef MNN_GPU_FORCE_FINISH
        glFinish();
#endif
    }

    return NO_ERROR;
}

ErrorCode GLConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    GPUConvolution::onResize(inputs, outputs);
    auto extra = (GLBackend *)backend();
    std::vector<std::string> prefix;
    GLint maxLocalSizeZ;
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &maxLocalSizeZ);
    if (mCommon->relu()) {
        prefix.push_back("#define RELU");
    }
    if (mCommon->relu6()) {
        prefix.push_back("#define RELU6");
    }

    auto dstDepthQuad = UP_DIV(outputs[0]->channel(), 4);
    mLocalSize[0]     = 1;
    mLocalSize[1]     = 1;
    mLocalSize[2]     = maxLocalSizeZ;
    if (mLocalSize[2] > dstDepthQuad) {
        mLocalSize[2] = dstDepthQuad;
    }

    // MNN_PRINT("%d, %d, %d for %d, %d, %d\n", mLocalSize[0], mLocalSize[1], mLocalSize[2], outputs[0]->width(),
    // outputs[0]->height(), outputs[0]->channel());

    {
        std::ostringstream os;
        os << "#define XLOCAL " << mLocalSize[0];
        prefix.push_back(os.str());
    }
    {
        std::ostringstream os;
        os << "#define YLOCAL " << mLocalSize[1];
        prefix.push_back(os.str());
    }
    {
        std::ostringstream os;
        os << "#define ZLOCAL " << mLocalSize[2];
        prefix.push_back(os.str());
    }
    if (1 == mCommon->kernelY() && 1 == mCommon->kernelX() && 1 == mCommon->strideY() && 1 == mCommon->strideX() &&
        0 == mCommon->padX() && 0 == mCommon->padY()) {
        mIs1x1      = true;
        mSetUniform = []() {};
    }

    if (mIs1x1) {
        mProgram = extra->getProgram("convolution1x1", glsl_convolution1x1_glsl, prefix);
    } else {
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
        mProgram = extra->getProgram("convolution", glsl_convolution_glsl, prefix);
    }

    return NO_ERROR;
}

} // namespace MNN

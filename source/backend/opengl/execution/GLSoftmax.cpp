//
//  GLSoftmax.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GLSoftmax.hpp"
#include <sstream>
#include "AllShader.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {
namespace OpenGL {
GLSoftmax::GLSoftmax(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {
    const auto softmaxParam = op->main_as_Axis();
    mAxis                   = softmaxParam->axis();
    mGLBackend = (GLBackend *)bn;

}

GLSoftmax::~GLSoftmax() {
    
}
    
ErrorCode GLSoftmax::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    std::vector<std::string> prefix;
    setLocalSize(prefix, mLocalSize, 8, 8, 1);
    if (1 == mAxis) { //channel
       mProgram = mGLBackend->getProgram("softmax_channel", glsl_softmaxChannel_glsl, prefix);
    } else if (2 == mAxis) { //height
        mProgram = mGLBackend->getProgram("softmax_height", glsl_softmaxHeight_glsl, prefix);
    } else if (3 == mAxis) { //width
        mProgram = mGLBackend->getProgram("softmax_width", glsl_softmaxWidth_glsl, prefix);
    } else {
        MNN_ASSERT(false);
    }
    
    return NO_ERROR;
}

ErrorCode GLSoftmax::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    auto input  = inputs[0];
    auto output = outputs[0];
    const int iw        = std::max(1, input->width());
    const int ih       = std::max(1, input->height());
    const int ic       = std::max(1, input->channel());
    const int ib       = std::max(1, input->batch());
    const int ic_4 = UP_DIV(input->channel(), 4);
    auto inputFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    if (MNN_DATA_FORMAT_NHWC == inputFormat) {
        // for NHWC input
        MNN_PRINT("Not Support MNN_DATA_FORMAT_NHWC == inputFormat !!! \n");
    } else {
        // NC4HW4 input
        mProgram->useProgram();
        glBindImageTexture(0, output->deviceId(), 0, GL_TRUE, 0, GL_WRITE_ONLY, TEXTURE_FORMAT);
        OPENGL_CHECK_ERROR;
        {
            int texId = 0;
            glActiveTexture(GL_TEXTURE0 + texId);
            glUniform1i(1, texId);
            glBindTexture(GL_TEXTURE_3D, input->deviceId());
            OPENGL_CHECK_ERROR;
        }
        glUniform1i(2, iw);
        glUniform1i(3, ih);
        glUniform1i(4, ic);
        if (1 == mAxis) { //channel
            glDispatchCompute(UP_DIV(iw, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]), UP_DIV(ib, mLocalSize[2]));
        } else if (2 == mAxis) { //height
            glDispatchCompute(UP_DIV(iw, mLocalSize[0]), UP_DIV(1, mLocalSize[1]),  UP_DIV(ic_4*ib, mLocalSize[2]));
        } else if (3 == mAxis) { //width
            glDispatchCompute(UP_DIV(1, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]),  UP_DIV(ic_4*ib, mLocalSize[2]));
        } else {
            MNN_ASSERT(false);
        }
        
    }
    return NO_ERROR;
}
GLCreatorRegister<TypedCreator<GLSoftmax>> __softmax_op(OpType_Softmax);
} // namespace OpenGL
} // namespace MNN

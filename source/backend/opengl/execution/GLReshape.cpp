//
//  GLReshape.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GLReshape.hpp"
#include <sstream>
#include "AllShader.hpp"
#include "GLBackend.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {
namespace OpenGL {
GLReshape::GLReshape(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {

}

GLReshape::~GLReshape() {
    
}
    
ErrorCode GLReshape::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    std::vector<std::string> prefix;
    setLocalSize(prefix, mLocalSize, 8, 8, 1);
    mTempBuffer.reset(new GLSSBOBuffer(ROUND_UP(inputs[0]->size(), 4)));
    if (TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4) {
        mSrcProgram = ((GLBackend *)backend())->getProgram("src", glsl_download_glsl, prefix);
        mDstProgram = ((GLBackend *)backend())->getProgram("dst", glsl_upload_glsl, prefix);
    }else{
        MNN_ASSERT(false);
    }
    
    return NO_ERROR;
}

ErrorCode GLReshape::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    int ih = input->height() > 0 ? input->height() : 1;
    int iw = input->width() > 0 ? input->width() : 1;
    int ic = input->channel() > 0 ? input->channel() : 1 ;
    int ib = input->batch() > 0 ? input->batch() : 1;
    int ic_4 = UP_DIV(ic, 4);
    
    int oh = output->height() > 0 ? output->height() : 1;
    int ow = output->width() > 0 ? output->width() : 1;
    int oc = output->channel() > 0 ? output->channel() : 1;
    int ob = output->batch() > 0 ? output->batch() : 1;
    int oc_4 = UP_DIV(oc, 4);

    printf("%d, %d \n", inputs[0]->size(), outputs[0]->size());
    MNN_PRINT("[%d, %d, %d, %d] -> [%d, %d, %d, %d] \n", ib, ic, ih, iw, ob, oc, oh, ow);
    //image -> buffer(nchw)
    {
        mSrcProgram->useProgram();
        glBindImageTexture(0, input->deviceId(), 0, GL_TRUE, 0, GL_READ_ONLY, TEXTURE_FORMAT);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mTempBuffer->getId());
        glUniform1i(2, iw);
        glUniform1i(3, ih);
        OPENGL_CHECK_ERROR;
        glDispatchCompute(UP_DIV(iw, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]), UP_DIV(ic_4, mLocalSize[2]));
        OPENGL_CHECK_ERROR;
    }
    
    //buffer -> image
    {
        mDstProgram->useProgram();
        glBindImageTexture(0, output->deviceId(), 0, GL_TRUE, 0, GL_WRITE_ONLY, TEXTURE_FORMAT);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mTempBuffer->getId());
        glUniform1i(2, ow);
        glUniform1i(3, oh);
        OPENGL_CHECK_ERROR;
        glDispatchCompute(UP_DIV(ow, mLocalSize[0]), UP_DIV(oh, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2]));
        OPENGL_CHECK_ERROR;
    }
    
    return NO_ERROR;
}
GLCreatorRegister<TypedCreator<GLReshape>> __reshape_op(OpType_Reshape);
} // namespace OpenGL
} // namespace MNN

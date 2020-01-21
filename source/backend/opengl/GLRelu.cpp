//
//  GLRelu.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLRelu.hpp"
#include <sstream>
#include "AllShader.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenGL {
GLRelu::GLRelu(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {
    mType = op->type();
    if(!(mType == OpType_PReLU)){
        mSlope = op->main_as_Relu()->slope();
    }else{
        mOp = op;
    }
}

GLRelu::~GLRelu() {

}

ErrorCode GLRelu::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    std::vector<std::string> prefix;
    setLocalSize(prefix, mLocalSize, 8, 8, 1);
    if (OpType_ReLU == mType) {
        prefix.push_back("#define RELU");
        mProgram = ((GLBackend *)backend())->getProgram("relu", glsl_relu_glsl, prefix);
    }else if(OpType_ReLU6 == mType){
        prefix.push_back("#define RELU6");
        mProgram = ((GLBackend *)backend())->getProgram("relu", glsl_relu_glsl, prefix);
    }else if(OpType_PReLU == mType){
        int count = ALIGN_UP4(mOp->main_as_PRelu()->slope()->size());
        mSlopeBuffer = std::shared_ptr<GLSSBOBuffer>(new GLSSBOBuffer(sizeof(float) * count));
        float* slope = (float*)(mSlopeBuffer->map(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
        if(slope != nullptr){
            ::memset(slope, 0, count * sizeof(float));
            ::memcpy(slope, mOp->main_as_PRelu()->slope()->data(), mOp->main_as_PRelu()->slope()->size() * sizeof(float));
        }
        mSlopeBuffer->unmap();
        mProgram = ((GLBackend *)backend())->getProgram("prelu", glsl_preluWithChannel_glsl, prefix);
    }else{
        MNN_PRINT("not support !!!");
        return NOT_SUPPORT;
    }
    return NO_ERROR;
}

ErrorCode GLRelu::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    int iw = input->width();
    int ih = input->height();
    int ic_4 = UP_DIV(input->channel(), 4);
    int ib = input->batch();

    if(OpType_PReLU == mType){
        mProgram->useProgram();
        glBindImageTexture(0, output->deviceId(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
        {
            int texId = 0;
            glActiveTexture(GL_TEXTURE0 + texId);
            glUniform1i(1, texId);
            glBindTexture(GL_TEXTURE_3D, input->deviceId());
            OPENGL_CHECK_ERROR;
        }
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mSlopeBuffer->getId());
        glUniform4i(3, iw, ih, ic_4, ib);
        OPENGL_CHECK_ERROR;
        ((GLBackend *)backend())->compute(UP_DIV(iw, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]), UP_DIV(ic_4, mLocalSize[2]));
    }else{
        mProgram->useProgram();
        glBindImageTexture(0, output->deviceId(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
        {
            int texId = 0;
            glActiveTexture(GL_TEXTURE0 + texId);
            glUniform1i(1, texId);
            glBindTexture(GL_TEXTURE_3D, input->deviceId());
            OPENGL_CHECK_ERROR;
        }
        glUniform4i(2, iw, ih, ic_4, ib);
        glUniform1f(3, mSlope);
        OPENGL_CHECK_ERROR;
        ((GLBackend *)backend())->compute(UP_DIV(iw, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]), UP_DIV(ic_4, mLocalSize[2]));
    }

    return NO_ERROR;
}
GLCreatorRegister<TypedCreator<GLRelu>> __relu_op(OpType_ReLU);
GLCreatorRegister<TypedCreator<GLRelu>> __prelu_op(OpType_PReLU);
} // namespace OpenGL
} // namespace MNN

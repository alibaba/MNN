//
//  GLPermute.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLPermute.hpp"
#include <sstream>
#include "AllShader.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenGL {
GLPermute::GLPermute(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {

    auto newDim = op->main_as_Permute()->dims();
    for (int i = 0; i < newDim->size(); ++i) {
        mDims.push_back(newDim->data()[i]);
    }
}

GLPermute::~GLPermute() {

}

ErrorCode GLPermute::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    std::vector<std::string> prefix;
    setLocalSize(prefix, mLocalSize, 8, 8, 1);
    mSrcBuffer.reset(new GLSSBOBuffer(input->size()));
    mDstBuffer.reset(new GLSSBOBuffer(output->size()));

    mPermuteProgram = ((GLBackend *)backend())->getProgram("permute", glsl_permute_glsl, prefix);
    mSrcProgram = ((GLBackend *)backend())->getProgram("src", glsl_image_to_nchw_buffer_glsl, prefix);
    mDstProgram = ((GLBackend *)backend())->getProgram("dst", glsl_nchw_buffer_to_image_glsl, prefix);

    return NO_ERROR;
}

ErrorCode GLPermute::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    int ih = input->height();
    int iw = input->width();
    int ic = input->channel();
    int ib = input->batch();
    int ic_4 = UP_DIV(ic, 4);

    int oh = output->height();
    int ow = output->width();
    int oc = output->channel();
    int ob = output->batch();
    int oc_4 = UP_DIV(oc, 4);

    //image -> buffer(nchw)
    {
        mSrcProgram->useProgram();
        glBindImageTexture(0, input->deviceId(), 0, GL_TRUE, 0, GL_READ_ONLY, ((GLBackend *)backend())->getTextrueFormat());
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mSrcBuffer->getId());
        glUniform1i(2, iw);
        glUniform1i(3, ih);
        OPENGL_CHECK_ERROR;
        ((GLBackend *)backend())->compute(UP_DIV(iw, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]), UP_DIV(ic_4, mLocalSize[2]));
        OPENGL_CHECK_ERROR;
    }

    //do permute
    {
        mPermuteProgram->useProgram();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mSrcBuffer->getId());
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mDstBuffer->getId());
        glUniform4i(2, mDims[0], mDims[1], mDims[2], mDims[3]);
        glUniform4i(3, iw, ih, ic, ib);
        glUniform4i(4, ow, oh, oc, ob);
        OPENGL_CHECK_ERROR;
        ((GLBackend *)backend())->compute(UP_DIV(ow, mLocalSize[0]), UP_DIV(oh, mLocalSize[1]), UP_DIV(oc, mLocalSize[2]));
        OPENGL_CHECK_ERROR;
    }

    //buffer(nchw) -> image
    {
        mDstProgram->useProgram();
        glBindImageTexture(0, output->deviceId(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mDstBuffer->getId());
        glUniform1i(2, ow);
        glUniform1i(3, oh);
        OPENGL_CHECK_ERROR;
        ((GLBackend *)backend())->compute(UP_DIV(ow, mLocalSize[0]), UP_DIV(oh, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2]));
        OPENGL_CHECK_ERROR;
    }

    return NO_ERROR;
}
GLCreatorRegister<TypedCreator<GLPermute>> __permute_op(OpType_Permute);
} // namespace OpenGL
} // namespace MNN

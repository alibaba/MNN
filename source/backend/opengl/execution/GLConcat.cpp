//
//  GLConcat.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GLConcat.hpp"
#include "AllShader.hpp"
#include "GLBackend.hpp"
#include "Macro.h"
namespace MNN {
namespace OpenGL {
GLConcat::~GLConcat() {
}

ErrorCode GLConcat::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto outputTensor = outputs[0];
    int dx            = 0;
    int dy            = 0;
    int dz            = 0;
    for (int i = 0; i < inputs.size(); ++i) {
        auto inputTensor = inputs[i];

        int sx = inputTensor->width();
        int sy = inputTensor->height();
        int sz = UP_DIV(inputTensor->channel(), 4);

        mProgram->useProgram();
        glBindImageTexture(0, (GLuint)outputTensor->deviceId(), 0, GL_TRUE, 0, GL_WRITE_ONLY, TEXTURE_FORMAT);

        glBindImageTexture(1, (GLuint)inputTensor->deviceId(), 0, GL_TRUE, 0, GL_READ_ONLY, TEXTURE_FORMAT);

        OPENGL_CHECK_ERROR;

        glUniform3i(2, 0, 0, 0);
        glUniform3i(3, dx, dy, dz);
        glUniform3i(4, sx, sy, sz);

        OPENGL_CHECK_ERROR;

        glDispatchCompute(UP_DIV(sx, 4), UP_DIV(sy, 4), UP_DIV(sz, 4));
        OPENGL_CHECK_ERROR;

        if (sx != outputTensor->width()) {
            dx += sx;
        } else if (sy != outputTensor->height()) {
            dy += sy;
        } else {
            dz += sz;
        }
    }
#ifdef MNN_GPU_FORCE_FINISH
    glFinish();
#endif

    return NO_ERROR;
}

GLConcat::GLConcat(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn): Execution(bn) {

    auto axis = op->main_as_Axis()->axis();
    bool valid = true;
    if (axis == 1) {
        for (int i = 1; i < inputs.size(); ++i) {
            if (inputs[i]->channel() % 4 != 0) {
                valid = false;
                break;
            }
        }
    }
    MNN_ASSERT(valid);
    if (valid) {
        mAxis    = axis;
    }
    mProgram = ((GLBackend *)backend())->getProgram("blit", glsl_blit_glsl);
}
GLCreatorRegister<TypedCreator<GLConcat>> __concat_op(OpType_Concat);
} // namespace OpenGL
} // namespace MNN

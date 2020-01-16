//
//  GLPool.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLPool.hpp"
#include "AllShader.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "core/Macro.h"
namespace MNN {
namespace OpenGL {
ErrorCode GLPool::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto inputTensor = inputs[0];
    auto pool        = mPool;
    if (!pool->isGlobal()) {
        int kx      = pool->kernelX();
        int ky      = pool->kernelY();
        int sx      = pool->strideX();
        int sy      = pool->strideY();
        int px      = pool->padX();
        int py      = pool->padY();
        mSetUniform = [=] {
            glUniform2i(2, kx, ky);
            glUniform2i(3, sx, sy);
            glUniform2i(4, px, py);
        };
    } else {
        mSetUniform = [=] {
            glUniform2i(2, inputTensor->width(), inputTensor->height());
            glUniform2i(3, 1, 1);
            glUniform2i(4, 0, 0);
        };
    }
    return NO_ERROR;
}

ErrorCode GLPool::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto imageInput   = inputs[0]->deviceId();
    auto imageOutput  = outputs[0]->deviceId();
    auto outputTensor = outputs[0];
    auto inputTensor  = inputs[0];

    MNN_ASSERT(mPoolProgram.get() != NULL);

    mPoolProgram->useProgram();
    glBindImageTexture(0, imageInput, 0, GL_TRUE, 0, GL_READ_ONLY, ((GLBackend *)backend())->getTextrueFormat());
    glBindImageTexture(1, imageOutput, 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
    mSetUniform();
    glUniform3i(10, outputTensor->width(), outputTensor->height(), UP_DIV(outputTensor->channel(), 4));
    glUniform3i(11, inputTensor->width(), inputTensor->height(), UP_DIV(inputTensor->channel(), 4));

    OPENGL_CHECK_ERROR;

    auto depthQuad = UP_DIV(outputTensor->channel(), 4);
    ((GLBackend *)backend())->compute(UP_DIV(outputTensor->width(), 2), UP_DIV(outputTensor->height(), 2), UP_DIV(depthQuad, 16));

    OPENGL_CHECK_ERROR;

    return NO_ERROR;
}

GLPool::~GLPool() {
}

GLPool::GLPool(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {
    auto extra = (GLBackend *)bn;
    auto pool = op->main_as_Pool();
    switch (pool->type()) {
        case PoolType_MAXPOOL:
            mPoolProgram = extra->getProgram("maxPool", glsl_maxpool_glsl);
            break;
        case PoolType_AVEPOOL:
            mPoolProgram = extra->getProgram("meanPool", glsl_avgpool_glsl);
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    mPool = pool;
}
GLCreatorRegister<TypedCreator<GLPool>> __pool_op(OpType_Pooling);
} // namespace OpenGL
} // namespace MNN

//
//  GLEltwise.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLEltwise.hpp"
#include <sstream>
#include "AllShader.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "core/Macro.h"
namespace MNN {
namespace OpenGL {
GLEltwise::GLEltwise(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {
    auto operation = op->main_as_Eltwise()->type();
    int inputCount = inputs.size();
    auto extra = (GLBackend *)bn;
    std::ostringstream shader;
    shader << "#define MAINOP(pos) ";
    auto inputNumber = inputCount;
    switch (operation) {
        case EltwiseType_MAXIMUM:
            for (int i = 0; i < inputNumber - 1; ++i) {
                shader << "max(imageLoad(uInput" << i << ", pos), ";
            }
            shader << "imageLoad(uInput" << (inputNumber - 1) << ", pos)";
            for (int i = 0; i < inputNumber - 1; ++i) {
                shader << ")";
            }

            break;
        case EltwiseType_PROD:
            shader << "imageLoad(uInput0, pos)";
            for (int i = 1; i < inputNumber; ++i) {
                shader << "*"
                       << "imageLoad(uInput" << i << ", pos)";
            }
            break;
        case EltwiseType_SUM:
            shader << "imageLoad(uInput0, pos)";
            for (int i = 1; i < inputNumber; ++i) {
                shader << "+"
                       << "imageLoad(uInput" << i << ", pos)";
            }
            break;
        default:
            break;
    }
    shader << "\n";
    for (int i = 0; i < inputNumber; ++i) {
        shader << "layout(FORMAT, binding=" << (i + 2) << ") readonly uniform highp image3D uInput" << i << ";\n";
    }
    shader << glsl_eltwise_glsl;
    mProgram = extra->getProgram("", shader.str().c_str());
}

GLEltwise::~GLEltwise() {
}

ErrorCode GLEltwise::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto outputTensor  = outputs[0];
    auto outputTexture = outputTensor->deviceId();

    mProgram->useProgram();
    glBindImageTexture(1, outputTexture, 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
    glUniform3i(10, outputTensor->width(), outputTensor->height(), UP_DIV(outputTensor->channel(), 4));
    OPENGL_CHECK_ERROR;

    for (int i = 0; i < inputs.size(); ++i) {
        auto inputTensor  = inputs[i];
        auto inputTexture = inputTensor->deviceId();
        glBindImageTexture(2 + i, inputTexture, 0, GL_TRUE, 0, GL_READ_ONLY, ((GLBackend *)backend())->getTextrueFormat());
    }
    auto depthQuad = UP_DIV(outputTensor->channel(), 4);
    ((GLBackend *)backend())->compute(UP_DIV(outputTensor->width(), 2), UP_DIV(outputTensor->height(), 2), UP_DIV(depthQuad, 16));
    OPENGL_CHECK_ERROR;

    return NO_ERROR;
}
GLCreatorRegister<TypedCreator<GLEltwise>> __eltwise_op(OpType_Eltwise);
} // namespace OpenGL
} // namespace MNN

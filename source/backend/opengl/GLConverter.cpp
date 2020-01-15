//
//  GLConverter.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLConverter.hpp"
#include "AllShader.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenGL {

GLConverter::GLConverter(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {
}

GLConverter::~GLConverter() {
}

ErrorCode GLConverter::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    std::vector<std::string> prefix;
    setLocalSize(prefix, mLocalSize, 8, 8, 1);
    mProgram = ((GLBackend *)backend())->getProgram("convert", glsl_converter_glsl, prefix);
    return NO_ERROR;
}

ErrorCode GLConverter::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);

    int ib = inputShape.at(0);
    int ih = inputShape.at(1);
    int iw = inputShape.at(2);
    int ic = inputShape.at(3);
    int ic_4 = UP_DIV(ic, 4);

    mProgram->useProgram();
    glBindImageTexture(0, output->deviceId(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
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
    glUniform1i(4, ic_4);

    ((GLBackend *)backend())->compute(UP_DIV(iw, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]),
                      UP_DIV(ic_4, mLocalSize[2]));
    return NO_ERROR;
}


GLCreatorRegister<TypedCreator<GLConverter>> __converter_op(OpType_ConvertTensor);
} // namespace OpenGL
} // namespace MNN

//
//  GLSqueeze.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLSqueeze.hpp"
#include <sstream>
#include "AllShader.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenGL {
GLSqueeze::GLSqueeze(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {

}

GLSqueeze::~GLSqueeze() {

}

ErrorCode GLSqueeze::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    std::vector<std::string> prefix;
    setLocalSize(prefix, mLocalSize, 8, 8, 1);
    mProgram = ((GLBackend *)backend())->getProgram("suqeeze", glsl_image_copy_glsl, prefix);

    return NO_ERROR;
}

ErrorCode GLSqueeze::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
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
    {
        int texId = 0;
        glActiveTexture(GL_TEXTURE0 + texId);
        glUniform1i(1, texId);
        glBindTexture(GL_TEXTURE_3D, input->deviceId());
        OPENGL_CHECK_ERROR;
    }
    glUniform4i(2, iw, ih, ic_4, ib);
    OPENGL_CHECK_ERROR;
    ((GLBackend *)backend())->compute(UP_DIV(iw, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]), UP_DIV(ic_4, mLocalSize[2]));

    return NO_ERROR;
}
class SqueezeCreator : public GLBackend::Creator {
public:
    virtual ~SqueezeCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {

        if(inputs[0]->dimensions() == 3 || outputs[0]->dimensions() == 3){
            MNN_PRINT("reshape not support dimensions == 3 \n");
            return nullptr;
        }
        return new GLSqueeze(inputs, op, backend);
    }
};
GLCreatorRegister<SqueezeCreator> __squeeze_op(OpType_Squeeze);
} // namespace OpenGL
} // namespace MNN

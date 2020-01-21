//
//  GLUnary.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLUnary.hpp"
#include <sstream>
#include "AllShader.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenGL {
GLUnary::GLUnary(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {
    mType = op->main_as_UnaryOp()->opType();
}

GLUnary::~GLUnary() {

}

ErrorCode GLUnary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    std::vector<std::string> prefix;
    setLocalSize(prefix, mLocalSize, 8, 8, 1);

    if (UnaryOpOperation_EXP == mType) {
        prefix.push_back("#define EXP");
        mProgram = ((GLBackend *)backend())->getProgram("unary_exp", glsl_unary_glsl, prefix);
    }else{
        MNN_PRINT("Not Supported Unary Operation: %d\n", mType);
    }
    return NO_ERROR;
}

ErrorCode GLUnary::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    auto input = inputs[0];
    auto output  = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

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
    glUniform4i(3, iw, ih, ic_4, 1);
    OPENGL_CHECK_ERROR;
    ((GLBackend *)backend())->compute(UP_DIV(iw, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]), UP_DIV(ic_4, mLocalSize[2]));

    return NO_ERROR;
}

class UnaryCreator : public GLBackend::Creator {
public:
    virtual ~UnaryCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto type = op->main_as_UnaryOp()->opType();
        if (UnaryOpOperation_EXP == type) {
            return new GLUnary(inputs, op, backend);
        }else{
            MNN_PRINT("Not Supported Unary Operation: %d\n", type);
            return nullptr;
        }

    }
};
GLCreatorRegister<UnaryCreator> __unary_op(OpType_UnaryOp);
} // namespace OpenGL
} // namespace MNN

//
//  GLBinary.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLBinary.hpp"
#include <sstream>
#include "AllShader.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenGL {
GLBinary::GLBinary(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {
    mType = op->main_as_BinaryOp()->opType();
    mActivationType = op->main_as_BinaryOp()->activationType();
}

GLBinary::~GLBinary() {

}

ErrorCode GLBinary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    std::vector<std::string> prefix;
    setLocalSize(prefix, mLocalSize, 8, 8, 1);
    if (BinaryOpOperation_ADD == mType) {
        prefix.push_back("#define ADD");
        mProgram = ((GLBackend *)backend())->getProgram("binary_add", glsl_binary_glsl, prefix);
    }else if(BinaryOpOperation_MUL == mType) {
        prefix.push_back("#define MUL");
        mProgram = ((GLBackend *)backend())->getProgram("binary_mul", glsl_binary_glsl, prefix);
    }else if(BinaryOpOperation_SUB == mType) {
        prefix.push_back("#define SUB");
        mProgram = ((GLBackend *)backend())->getProgram("binary_sub", glsl_binary_glsl, prefix);
    }else if(BinaryOpOperation_REALDIV == mType) {
        prefix.push_back("#define REALDIV");
        mProgram = ((GLBackend *)backend())->getProgram("binary_realdiv", glsl_binary_glsl, prefix);
    }else{
        MNN_PRINT("Not Supported Binary Operation: %d\n", mType);
    }
    return NO_ERROR;
}

ErrorCode GLBinary::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(2 == inputs.size());
    MNN_ASSERT(1 == outputs.size());

    auto input0 = inputs[0];
    auto input1 = inputs[1];
    auto output  = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input0);
    std::vector<int> outputShape = tensorShapeFormat(output);

    int ib = inputShape.at(0);
    int ih = inputShape.at(1);
    int iw = inputShape.at(2);
    int ic = inputShape.at(3);
    int ic_4 = UP_DIV(ic, 4);

    int ob = outputShape.at(0);
    int oh = outputShape.at(1);
    int ow = outputShape.at(2);
    int oc = outputShape.at(3);
    int oc_4 = UP_DIV(oc, 4);

    MNN_ASSERT(input0->getType().code == halide_type_float);
    MNN_ASSERT(input0->dimensions() == input1->dimensions());

    const auto intputFormat = TensorUtils::getDescribe(input0)->dimensionFormat;

    mProgram->useProgram();
    glBindImageTexture(0, output->deviceId(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
    {
        int texId = 0;
        glActiveTexture(GL_TEXTURE0 + texId);
        glUniform1i(1, texId);
        glBindTexture(GL_TEXTURE_3D, input0->deviceId());
        OPENGL_CHECK_ERROR;
    }
    {
        int texId = 1;
        glActiveTexture(GL_TEXTURE0 + texId);
        glUniform1i(2, texId);
        glBindTexture(GL_TEXTURE_3D, input1->deviceId());
        OPENGL_CHECK_ERROR;
    }
    glUniform4i(3, iw, ih, ic_4, 1);
    glUniform1i(4, mActivationType);
    OPENGL_CHECK_ERROR;
    ((GLBackend *)backend())->compute(UP_DIV(iw, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]), UP_DIV(ic_4, mLocalSize[2]));

    return NO_ERROR;
}

class BinaryCreator : public GLBackend::Creator {
public:
    virtual ~BinaryCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {

        MNN_ASSERT(inputs.size() > 1);
        auto input0 = inputs[0];
        // Don't support broatcast
        for (int i = 1; i < inputs.size(); ++i) {
            auto input = inputs[i];
            if (input0->dimensions() != input->dimensions()) {
                MNN_PRINT("dimensions : [%d, %d] \n", input0->dimensions(), input->dimensions());
                MNN_PRINT("opengl binary don't support broatcast !!! \n");
                return nullptr;
            }
            auto dim = input0->dimensions();
            for (int l = 0; l < dim; ++l) {
                if (input0->length(l) != input->length(l)) {
                    MNN_PRINT("length : [%d, %d] \n", input0->length(l), input->length(l));
                    MNN_PRINT("opengl binary don't support broatcast !!! \n");
                    return nullptr;
                }
            }
        }

        return new GLBinary(inputs, op, backend);
    }
};
GLCreatorRegister<BinaryCreator> __binary_op(OpType_BinaryOp);
} // namespace OpenGL
} // namespace MNN

//
//  GLBinary.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GLBinary.hpp"
#include <sstream>
#include "AllShader.hpp"
#include "GLBackend.hpp"
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {
namespace OpenGL {
GLBinary::GLBinary(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {
    mType = op->main_as_BinaryOp()->opType();
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
        mProgram = ((GLBackend *)backend())->getProgram("binary_mul", glsl_binary_glsl, prefix);;
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
    if (intputFormat == MNN_DATA_FORMAT_NHWC) {
        MNN_PRINT("NOT SUPPORT MNN_DATA_FORMAT_NHWC !!!\n");
    }else if(intputFormat == MNN_DATA_FORMAT_NC4HW4){
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
        OPENGL_CHECK_ERROR;
        ((GLBackend *)backend())->compute(UP_DIV(iw, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]), UP_DIV(ic_4, mLocalSize[2]));
    }

    return NO_ERROR;
}
GLCreatorRegister<TypedCreator<GLBinary>> __binary_op(OpType_BinaryOp);
} // namespace OpenGL
} // namespace MNN

//
//  GLSoftmax.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLSoftmax.hpp"
#include <sstream>
#include "AllShader.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenGL {
GLSoftmax::GLSoftmax(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {
    const auto softmaxParam = op->main_as_Axis();
    mAxis                   = softmaxParam->axis();
    mGLBackend = (GLBackend *)bn;

}

GLSoftmax::~GLSoftmax() {

}

ErrorCode GLSoftmax::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    std::vector<std::string> prefix;
    setLocalSize(prefix, mLocalSize, 8, 8, 1);
    if (mAxis < 0) {
        mAxis = inputs[0]->dimensions() + mAxis;
    }
    if (1 == mAxis) { //channel
        mProgram = mGLBackend->getProgram("softmax_channel", glsl_softmaxChannel_glsl, prefix);
    } else if (2 == mAxis) { //height
        mProgram = mGLBackend->getProgram("softmax_height", glsl_softmaxHeight_glsl, prefix);
    } else if (3 == mAxis) { //width
        mProgram = mGLBackend->getProgram("softmax_width", glsl_softmaxWidth_glsl, prefix);
    } else {
        MNN_ASSERT(false);
    }
    return NO_ERROR;
}

ErrorCode GLSoftmax::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    auto input  = inputs[0];
    auto output = outputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
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

    // NC4HW4 input
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
    glUniform1i(4, ic);
    if (1 == mAxis) { //channel
        ((GLBackend *)backend())->compute(UP_DIV(iw, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]), UP_DIV(ib, mLocalSize[2]));
    } else if (2 == mAxis) { //height
        ((GLBackend *)backend())->compute(UP_DIV(iw, mLocalSize[0]), UP_DIV(1, mLocalSize[1]),  UP_DIV(ic_4*ib, mLocalSize[2]));
    } else if (3 == mAxis) { //width
        ((GLBackend *)backend())->compute(UP_DIV(1, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]),  UP_DIV(ic_4*ib, mLocalSize[2]));
    } else {
        MNN_ASSERT(false);
    }

    return NO_ERROR;
}

class SoftmaxCreator : public GLBackend::Creator {
public:
    virtual ~SoftmaxCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if(TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NHWC){
            MNN_PRINT("softmax not support dimensionFormat == MNN_DATA_FORMAT_NHWC \n");
            return nullptr;
        }else if(inputs[0]->dimensions() == 3 || outputs[0]->dimensions() == 3){
            MNN_PRINT("softmax not support dimensions == 3 \n");
            return nullptr;
        }
        return new GLSoftmax(inputs, op, backend);
    }
};
GLCreatorRegister<SoftmaxCreator> __softmax_op(OpType_Softmax);
} // namespace OpenGL
} // namespace MNN

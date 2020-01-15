//
//  GLReshape.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLReshape.hpp"
#include <sstream>
#include "AllShader.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenGL {
GLReshape::GLReshape(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {
    mDimType = op->main_as_Reshape()->dimType();
}

GLReshape::~GLReshape() {

}

ErrorCode GLReshape::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    std::vector<std::string> prefix;
    setLocalSize(prefix, mLocalSize, 8, 8, 1);
    mTempBuffer.reset(new GLSSBOBuffer(inputs[0]->size()));
    auto input = inputs[0];

    if (mDimType == MNN_DATA_FORMAT_NCHW) {
        mSrcProgram = ((GLBackend *)backend())->getProgram("src", glsl_image_to_nchw_buffer_glsl, prefix);
        mDstProgram = ((GLBackend *)backend())->getProgram("dst", glsl_nchw_buffer_to_image_glsl, prefix);
    }else{
        mSrcProgram = ((GLBackend *)backend())->getProgram("src", glsl_image_to_nhwc_buffer_glsl, prefix);
        mDstProgram = ((GLBackend *)backend())->getProgram("dst", glsl_nhwc_buffer_to_image_glsl, prefix);
    }

    return NO_ERROR;
}

ErrorCode GLReshape::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
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

    if (mDimType == MNN_DATA_FORMAT_NCHW) {
        //image -> buffer(nchw)
        {
            mSrcProgram->useProgram();
            glBindImageTexture(0, input->deviceId(), 0, GL_TRUE, 0, GL_READ_ONLY, ((GLBackend *)backend())->getTextrueFormat());
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mTempBuffer->getId());
            glUniform1i(2, iw);
            glUniform1i(3, ih);
            OPENGL_CHECK_ERROR;
            ((GLBackend *)backend())->compute(UP_DIV(iw, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]), UP_DIV(ic_4, mLocalSize[2]));
            OPENGL_CHECK_ERROR;
        }
        //buffer(nchw) -> image
        {
            mDstProgram->useProgram();
            glBindImageTexture(0, output->deviceId(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mTempBuffer->getId());
            glUniform1i(2, ow);
            glUniform1i(3, oh);
            OPENGL_CHECK_ERROR;
            ((GLBackend *)backend())->compute(UP_DIV(ow, mLocalSize[0]), UP_DIV(oh, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2]));
            OPENGL_CHECK_ERROR;
        }
    }else{
        //image -> buffer(nhwc)
        {
            mSrcProgram->useProgram();
            glBindImageTexture(0, input->deviceId(), 0, GL_TRUE, 0, GL_READ_ONLY, ((GLBackend *)backend())->getTextrueFormat());
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mTempBuffer->getId());
            glUniform1i(2, iw);
            glUniform1i(3, ih);
            glUniform1i(4, ic);
            OPENGL_CHECK_ERROR;
            ((GLBackend *)backend())->compute(UP_DIV(iw, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]), UP_DIV(ic_4, mLocalSize[2]));
            OPENGL_CHECK_ERROR;
        }

        //buffer -> image
        {
            mDstProgram->useProgram();
            glBindImageTexture(0, output->deviceId(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mTempBuffer->getId());
            glUniform1i(2, ow);
            glUniform1i(3, oh);
            glUniform1i(4, oc);
            OPENGL_CHECK_ERROR;
            ((GLBackend *)backend())->compute(UP_DIV(ow, mLocalSize[0]), UP_DIV(oh, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2]));
            OPENGL_CHECK_ERROR;
        }
    }

    return NO_ERROR;
}

class ReshapeCreator : public GLBackend::Creator {
public:
    virtual ~ReshapeCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {

        if(inputs[0]->dimensions() == 3 || outputs[0]->dimensions() == 3){
            MNN_PRINT("reshape not support dimensions == 3 \n");
            return nullptr;
        }
        return new GLReshape(inputs, op, backend);
    }
};
GLCreatorRegister<ReshapeCreator> __reshape_op(OpType_Reshape);
} // namespace OpenGL
} // namespace MNN

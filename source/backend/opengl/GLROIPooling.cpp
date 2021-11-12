//
//  GLRoiPooling.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLROIPooling.hpp"
#include "AllShader.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "core/Macro.h"
namespace MNN {
namespace OpenGL {
GLRoiPooling::GLRoiPooling(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {
    auto extra = (GLBackend *)bn;
    std::vector<std::string> prefix;
    setLocalSize(prefix, mLocalSize, 8, 8, 1);
    mPoolProgram = extra->getProgram("roipooling", glsl_roiPooling_glsl, prefix);
    mSpatialScale = op->main_as_RoiParameters()->spatialScale();
}

GLRoiPooling::~GLRoiPooling() {
}


ErrorCode GLRoiPooling::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    return NO_ERROR;
}

ErrorCode GLRoiPooling::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto output = outputs[0];
    auto input  = inputs[0];
    auto roi  = inputs[0];

    int ob = output->batch();
    int oc = output->channel();
    int oh = output->height();
    int ow = output->width();
    int oc_4 = UP_DIV(oc, 4);

    mPoolProgram->useProgram();
    {
        int texId = 0;
        glActiveTexture(GL_TEXTURE0 + texId);
        glUniform1i(0, texId);
        glBindTexture(GL_TEXTURE_3D, input->deviceId());
        OPENGL_CHECK_ERROR;
    }
    glBindImageTexture(1, output->deviceId(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
    {
        int texId = 1;
        glActiveTexture(GL_TEXTURE0 + texId);
        glUniform1i(2, texId);
        glBindTexture(GL_TEXTURE_3D, roi->deviceId());
        OPENGL_CHECK_ERROR;
    }
    glUniform3i(10, output->width(), output->height(), UP_DIV(output->channel(), 4));
    glUniform3i(11, input->width(), input->height(), UP_DIV(input->channel(), 4));
    glUniform1f(12, mSpatialScale);

    OPENGL_CHECK_ERROR;

    ((GLBackend *)backend())->compute(UP_DIV(ow, mLocalSize[0]), UP_DIV(oh, mLocalSize[1]), UP_DIV(oc_4, mLocalSize[2]));

    OPENGL_CHECK_ERROR;

    return NO_ERROR;
}

GLCreatorRegister<TypedCreator<GLRoiPooling>> __roipooling_op(OpType_ROIPooling);
} // namespace OpenGL
} // namespace MNN

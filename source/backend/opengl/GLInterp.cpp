//
//  GLInterp.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLInterp.hpp"
#include <sstream>
#include "AllShader.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "core/Macro.h"
namespace MNN {
namespace OpenGL {
GLInterp::GLInterp(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {
    auto interpParam = op->main_as_Interp();
    mAlignCorners    = interpParam->alignCorners();
    mResizeType = interpParam->resizeType();
}

ErrorCode GLInterp::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    std::vector<std::string> prefix;
    setLocalSize(prefix, mLocalSize, 8, 8, 1);
    if(mResizeType == 1){
        mProgram = ((GLBackend *)backend())->getProgram("interp_nearest", glsl_resizeNearest_glsl, prefix);
    }else if(mResizeType == 2){
        mProgram = ((GLBackend *)backend())->getProgram("interp_bilinear", glsl_resizeBilinear_glsl, prefix);
    }else{
        return NOT_SUPPORT;
    }
    return NO_ERROR;
}

GLInterp::~GLInterp() {
}

ErrorCode GLInterp::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output  = outputs[0];

    int iw = input->width();
    int ih = input->height();
    int ic_4 = UP_DIV(input->channel(), 4);
    int ib = input->batch();

    int ow = output->width();
    int oh = output->height();
    int oc_4 = UP_DIV(output->channel(), 4);
    int ob = output->batch();

    float xScale = 1;
    float yScale = 1;
    if (mAlignCorners) {
        yScale = (float)(ih - 1) / (float)(oh - 1);
        xScale = (float)(iw - 1) / (float)(ow - 1);
    } else {
        yScale = (float)(ih) / (float)(oh);
        xScale = (float)(iw) / (float)(ow);
    }

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
    glUniform4i(3, ow, oh, oc_4, ob);
    glUniform2f(4, xScale, yScale);
    OPENGL_CHECK_ERROR;
    ((GLBackend *)backend())->compute(UP_DIV(ow, mLocalSize[0]), UP_DIV(oh, mLocalSize[1]), UP_DIV(oc_4*ob, mLocalSize[2]));

    return NO_ERROR;
}
GLCreatorRegister<TypedCreator<GLInterp>> __interp_op(OpType_Interp);
} // namespace OpenGL
} // namespace MNN

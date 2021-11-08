//
//  GLConcat.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLConcat.hpp"
#include "AllShader.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "core/Macro.h"
namespace MNN {
namespace OpenGL {
GLConcat::GLConcat(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn): Execution(bn) {
    mAxis    = op->main_as_Axis()->axis();
    mProgram = ((GLBackend *)backend())->getProgram("blit", glsl_blit_glsl);
}

GLConcat::~GLConcat() {
}

ErrorCode GLConcat::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto outputTensor = outputs[0];
    std::vector<int> outputShape  = tensorShapeFormat(outputTensor);
    int dx            = 0;
    int dy            = 0;
    int dz            = 0;
    for (int i = 0; i < inputs.size(); ++i) {
        auto inputTensor = inputs[i];

        std::vector<int> inputShape  = tensorShapeFormat(inputTensor);

        int sy = inputShape.at(1);
        int sx = inputShape.at(2);
        int ic = inputShape.at(3);

        int sz = UP_DIV(ic, 4);

        mProgram->useProgram();
        glBindImageTexture(0, (GLuint)outputTensor->deviceId(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());

        glBindImageTexture(1, (GLuint)inputTensor->deviceId(), 0, GL_TRUE, 0, GL_READ_ONLY, ((GLBackend *)backend())->getTextrueFormat());

        OPENGL_CHECK_ERROR;

        glUniform3i(2, 0, 0, 0);
        glUniform3i(3, dx, dy, dz);
        glUniform3i(4, sx, sy, sz);

        OPENGL_CHECK_ERROR;

        ((GLBackend *)backend())->compute(UP_DIV(sx, 4), UP_DIV(sy, 4), UP_DIV(sz, 4));

        OPENGL_CHECK_ERROR;

        if (sx != outputShape.at(2)) {
            dx += sx;
        } else if (sy != outputShape.at(1)) {
            dy += sy;
        } else {
            dz += sz;
        }
    }

    return NO_ERROR;
}

class ConcatCreator : public GLBackend::Creator {
public:
    virtual ~ConcatCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {

        auto axis = op->main_as_Axis()->axis();
        if (0 > axis) {
            axis = outputs[0]->dimensions() + axis;
        }
        for (int i = 0; i < inputs.size(); ++i) {
            if (inputs[i]->getDimensionType() != Tensor::CAFFE) {
                // TODO Support NHWC format
                return nullptr;
            }
        }
        if (axis == 1) {
            for (int i = 0; i < inputs.size() - 1; ++i) {
                if (inputs[i]->channel() % 4 != 0) {
                    MNN_PRINT("concat only support 4 alignment, back to cpu !!! \n");
                    return nullptr;
                }
            }
        }

        return new GLConcat(inputs, op, backend);
    }
};

GLCreatorRegister<ConcatCreator> __concat_op(OpType_Concat);
} // namespace OpenGL
} // namespace MNN

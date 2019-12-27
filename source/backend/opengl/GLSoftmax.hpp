//
//  GLSoftmax.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLSoftmax_H
#define GLSoftmax_H
#include "core/Execution.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLTexture.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace OpenGL {
class GLSoftmax : public MNN::Execution {
public:
    GLSoftmax(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn);
    virtual ~GLSoftmax();
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mProgram;
    int mLocalSize[3];
    int mAxis;
    GLBackend * mGLBackend;
};
} // namespace OpenGL
} // namespace MNN

#endif // GLSoftmax_H

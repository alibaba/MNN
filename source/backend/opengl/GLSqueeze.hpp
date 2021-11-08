//
//  GLSqueeze.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLSqueeze_H
#define GLSqueeze_H
#include "core/Execution.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLTexture.hpp"
#include "backend/opengl/GLSSBOBuffer.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace OpenGL {
class GLSqueeze : public MNN::Execution {
public:
    GLSqueeze(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn);
    virtual ~GLSqueeze();
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mProgram;
    int mLocalSize[3];
};
} // namespace OpenGL
} // namespace MNN

#endif // GLSqueeze_H

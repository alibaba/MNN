//
//  GLUnary.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLUnary_H
#define GLUnary_H
#include "core/Execution.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLTexture.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace OpenGL {
class GLUnary : public MNN::Execution {
public:
    GLUnary(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn);
    virtual ~GLUnary();
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mProgram;
    int32_t mType;
    int mLocalSize[3];
};
} // namespace OpenGL
} // namespace MNN

#endif // GLUnary_H

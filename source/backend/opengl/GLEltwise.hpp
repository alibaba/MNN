//
//  GLEltwise.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDEMO_GLELTWISE_H
#define MNNDEMO_GLELTWISE_H
#include "core/Execution.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLTexture.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace OpenGL {
class GLEltwise : public MNN::Execution {
public:
    GLEltwise(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn);
    virtual ~GLEltwise();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mProgram;
};
} // namespace OpenGL
} // namespace MNN

#endif // MNNDEMO_GLELTWISE_H

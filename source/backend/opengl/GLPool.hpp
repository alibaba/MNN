//
//  GLPool.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDEMO_GLPOOL_H
#define MNNDEMO_GLPOOL_H

#include "core/Execution.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLTexture.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace OpenGL {
class GLPool : public MNN::Execution {
public:
    GLPool(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn);
    virtual ~GLPool();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mPoolProgram;
    const Pool *mPool;
    std::function<void()> mSetUniform;
};
} // namespace OpenGL
} // namespace MNN

#endif // MNNDEMO_GLPOOL_H

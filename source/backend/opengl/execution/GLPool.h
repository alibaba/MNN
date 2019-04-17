//
//  GLPool.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDEMO_GLPOOL_H
#define MNNDEMO_GLPOOL_H

#include "Execution.hpp"
#include "GLProgram.h"
#include "GLTexture.h"
#include "MNN_generated.h"
namespace MNN {
class GLPool : public MNN::Execution {
public:
    GLPool(const Pool *pool, Backend *bn);
    virtual ~GLPool();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mPoolProgram;
    const Pool *mPool;
    std::function<void()> mSetUniform;
};
} // namespace MNN

#endif // MNNDEMO_GLPOOL_H

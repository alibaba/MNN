//
//  GLEltwise.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDEMO_GLELTWISE_H
#define MNNDEMO_GLELTWISE_H
#include "Execution.hpp"
#include "GLProgram.h"
#include "GLTexture.h"
#include "MNN_generated.h"
namespace MNN {
class GLEltwise : public MNN::Execution {
public:
    GLEltwise(EltwiseType operation, int inputCount, Backend *bn);
    virtual ~GLEltwise();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mProgram;
};
} // namespace MNN

#endif // MNNDEMO_GLELTWISE_H

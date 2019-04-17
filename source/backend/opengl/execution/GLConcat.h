//
//  GLConcat.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDEMO_GLCONCAT_H
#define MNNDEMO_GLCONCAT_H

#include "Execution.hpp"
#include "GLProgram.h"
#include "GLTexture.h"
namespace MNN {
class GLConcat : public Execution {
public:
    GLConcat(int axis, Backend *bn);
    virtual ~GLConcat();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mProgram;
    int mAxis;
};
} // namespace MNN

#endif // MNNDEMO_GLCONCAT_H

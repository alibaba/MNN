//
//  GLConcat.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_GLCONCAT_H
#define MNN_GLCONCAT_H

#include "core/Execution.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLTexture.hpp"
namespace MNN {
namespace OpenGL {
class GLConcat : public Execution {
public:
    GLConcat(const std::vector<Tensor *> &inputs, const Op *convOp, Backend *bn);
    virtual ~GLConcat();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mProgram;
    int mAxis;
};
} // namespace OpenGL
} // namespace MNN

#endif // MNN_GLCONCAT_H

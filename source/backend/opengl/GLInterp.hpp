//
//  GLInterp.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLInterp_H
#define GLInterp_H
#include "core/Execution.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLTexture.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace OpenGL {
class GLInterp : public MNN::Execution {
public:
    GLInterp(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn);
    virtual ~GLInterp();
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mProgram;
    bool mAlignCorners;
    int mResizeType;
    int mLocalSize[3];
};
} // namespace OpenGL
} // namespace MNN

#endif // GLInterp_H

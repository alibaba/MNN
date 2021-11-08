//
//  GLPermute.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLPermute_H
#define GLPermute_H
#include "core/Execution.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLTexture.hpp"
#include "backend/opengl/GLSSBOBuffer.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace OpenGL {
class GLPermute : public MNN::Execution {
public:
    GLPermute(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn);
    virtual ~GLPermute();
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mPermuteProgram;
    std::shared_ptr<GLProgram> mSrcProgram;
    std::shared_ptr<GLProgram> mDstProgram;
    std::shared_ptr<GLSSBOBuffer> mSrcBuffer;
    std::shared_ptr<GLSSBOBuffer> mDstBuffer;
    int mLocalSize[3];
    std::vector<int> mDims;
};
} // namespace OpenGL
} // namespace MNN

#endif // GLPermute_H

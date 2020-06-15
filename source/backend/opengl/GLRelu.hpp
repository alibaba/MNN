//
//  GLRelu.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLRelu_H
#define GLRelu_H
#include "core/Execution.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLTexture.hpp"
#include "backend/opengl/GLSSBOBuffer.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace OpenGL {
class GLRelu : public MNN::Execution {
public:
    GLRelu(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn);
    virtual ~GLRelu();
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mProgram;
    std::shared_ptr<GLSSBOBuffer> mSlopeBuffer;
    int mLocalSize[3];
    int mType;
    float mSlope;
    const Op* mOp;
};
} // namespace OpenGL
} // namespace MNN

#endif // GLRelu_H

//
//  GLReshape.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLReshape_H
#define GLReshape_H
#include "core/Execution.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLTexture.hpp"
#include "backend/opengl/GLSSBOBuffer.hpp"
#include "MNN_generated.h"

namespace MNN {
namespace OpenGL {
class GLReshape : public MNN::Execution {
public:
    GLReshape(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn);
    virtual ~GLReshape();
    ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mSrcProgram;
    std::shared_ptr<GLProgram> mDstProgram;
    std::shared_ptr<GLSSBOBuffer> mTempBuffer;
    int mLocalSize[3];
    MNN_DATA_FORMAT mDimType;
};
} // namespace OpenGL
} // namespace MNN

#endif // GLReshape_H

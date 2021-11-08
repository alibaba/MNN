//
//  GLRoiPooling.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLRoiPooling_H
#define GLRoiPooling_H

#include "core/Execution.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLTexture.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace OpenGL {
class GLRoiPooling : public MNN::Execution {
public:
    GLRoiPooling(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn);
    virtual ~GLRoiPooling();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mPoolProgram;
    std::function<void()> mSetUniform;
    int mLocalSize[3];
    float mSpatialScale;
};
} // namespace OpenGL
} // namespace MNN

#endif // GLRoiPooling_H

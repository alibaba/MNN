//
//  GLConverter.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDEMO_GLConverter_H
#define MNNDEMO_GLConverter_H

#include "core/Execution.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLTexture.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace OpenGL {
class GLConverter : public MNN::Execution {
public:
    GLConverter(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn);
    virtual ~GLConverter();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<GLProgram> mProgram;
    int mLocalSize[3];
};
} // namespace OpenGL
} // namespace MNN

#endif // MNNDEMO_GLConverter_H

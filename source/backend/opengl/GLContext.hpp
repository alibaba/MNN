//
//  GLContext.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLCONTEXT_H
#define GLCONTEXT_H

#include "backend/opengl/GLHead.hpp"
#include <EGL/egl.h>
#include <string>
#include <unordered_set>
namespace MNN {
namespace OpenGL {
class GLContext {
public:
    GLContext();
    ~GLContext();
    bool isCreateError() const;
private:
    EGLContext mContext;
    EGLDisplay mDisplay;
    EGLSurface mSurface;
    bool mIsCreateError{false};
};
} // namespace OpenGL
} // namespace MNN

#endif

//
//  GLContext.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GLContext.hpp"
#include <EGL/egl.h>
namespace MNN {
namespace OpenGL {
class GLContext::nativeContext {
public:
    nativeContext() {
        mDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        EGLint majorVersion;
        EGLint minorVersion;
        eglInitialize(mDisplay, &majorVersion, &minorVersion);
        EGLint numConfigs;
        static const EGLint configAttribs[] = {EGL_SURFACE_TYPE,
                                               EGL_PBUFFER_BIT,
                                               EGL_RENDERABLE_TYPE,
                                               EGL_OPENGL_ES2_BIT,
                                               EGL_RED_SIZE,
                                               8,
                                               EGL_GREEN_SIZE,
                                               8,
                                               EGL_BLUE_SIZE,
                                               8,
                                               EGL_ALPHA_SIZE,
                                               8,
                                               EGL_NONE};

        EGLConfig surfaceConfig;
        eglChooseConfig(mDisplay, configAttribs, &surfaceConfig, 1, &numConfigs);

        static const EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
        mContext                             = eglCreateContext(mDisplay, surfaceConfig, NULL, contextAttribs);

        static const EGLint surfaceAttribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
        mSurface                             = eglCreatePbufferSurface(mDisplay, surfaceConfig, surfaceAttribs);
        eglMakeCurrent(mDisplay, mSurface, mSurface, mContext);
    }
    ~nativeContext() {
        eglMakeCurrent(mDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(mDisplay, mContext);
        eglDestroySurface(mDisplay, mSurface);
        eglTerminate(mDisplay);
        mDisplay = EGL_NO_DISPLAY;
    }

private:
    EGLContext mContext;
    EGLDisplay mDisplay;
    EGLSurface mSurface;
};

GLContext::nativeContext* GLContext::create(int version) {
    return new nativeContext;
}

void GLContext::destroy(nativeContext* context) {
    if(context != nullptr){
        delete context;
    }
}
} // namespace OpenGL
} // namespace MNN

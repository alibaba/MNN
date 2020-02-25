//
//  GLContext.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLContext.hpp"
namespace MNN {
namespace OpenGL {
    GLContext::GLContext() {
        if(!(eglGetCurrentContext() != EGL_NO_CONTEXT)){
            mDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
            if (mDisplay == EGL_NO_DISPLAY) {
                MNN_PRINT("eglGetDisplay error !!! \n");
                mIsCreateError = true;
            }
            int majorVersion;
            int minorVersion;
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
            if(!eglChooseConfig(mDisplay, configAttribs, &surfaceConfig, 1, &numConfigs)){
                eglMakeCurrent(mDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
                eglTerminate(mDisplay);
                mDisplay = EGL_NO_DISPLAY;
                MNN_PRINT("eglChooseConfig error !!! \n");
                mIsCreateError = true;
            }

            static const EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
            mContext                             = eglCreateContext(mDisplay, surfaceConfig, NULL, contextAttribs);
            static const EGLint surfaceAttribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
            mSurface                             = eglCreatePbufferSurface(mDisplay, surfaceConfig, surfaceAttribs);
            eglMakeCurrent(mDisplay, mSurface, mSurface, mContext);
            eglBindAPI(EGL_OPENGL_ES_API);
            int major;
            glGetIntegerv(GL_MAJOR_VERSION, &major);
            if(major < 3){
                mIsCreateError = true;
            }
        }else{
            mContext = EGL_NO_CONTEXT;
            MNN_PRINT("eglGetCurrentContext() != EGL_NO_CONTEXT \n");
            mIsCreateError = true;
        }
    }
    GLContext::~GLContext() {
        if (mDisplay != EGL_NO_DISPLAY) {
            if (mContext != EGL_NO_CONTEXT) {
                eglDestroyContext(mDisplay, mContext);
                mContext = EGL_NO_CONTEXT;
            }
            if (mSurface != EGL_NO_SURFACE) {
                eglDestroySurface(mDisplay, mSurface);
                mSurface = EGL_NO_SURFACE;
            }
            eglMakeCurrent(mDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            eglTerminate(mDisplay);
            mDisplay = EGL_NO_DISPLAY;
        }
        eglReleaseThread();
    }
    bool GLContext::isCreateError() const{
        return mIsCreateError;
    }

} // namespace OpenGL
} // namespace MNN

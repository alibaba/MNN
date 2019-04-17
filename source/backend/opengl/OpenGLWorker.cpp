//
//  OpenGLWorker.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpenGLWorker.h"
#include <stdlib.h>
#include "GLContext.h"
namespace MNN {

OpenGLWorker* OpenGLWorker::gInstance = NULL;

static GLLock gLock;

GLWorkThread* OpenGLWorker::getInstance() {
    if (NULL == gInstance) {
        GLAutoLock _l(gLock);
        if (NULL == gInstance) {
            gInstance = new OpenGLWorker;
        }
    }
    GLASSERT(NULL != gInstance);
    return gInstance->mThread;
}

OpenGLWorker::OpenGLWorker() {
    mThread = new GLWorkThread();
    mThread->start();
}
OpenGLWorker::~OpenGLWorker() {
    mThread->stop();
    delete mThread;
}
} // namespace MNN

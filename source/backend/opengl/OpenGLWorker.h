//
//  OpenGLWorker.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OPENGLWORKER_H
#define OPENGLWORKER_H
#include "GLWorkThread.h"
namespace MNN {

class OpenGLWorker {
public:
    static GLWorkThread* getInstance();

private:
    GLWorkThread* mThread;
    OpenGLWorker();
    ~OpenGLWorker();
    static OpenGLWorker* gInstance;
};
} // namespace MNN
#endif

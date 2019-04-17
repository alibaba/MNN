//
//  GLOnceWork.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLONCEWORK_H
#define GLONCEWORK_H
#include <functional>
#include "GLThread.h"
class GLOnceWork : public GLThread {
public:
    GLOnceWork(std::function<void(void)>* f) : mF(f) {
    }
    virtual ~GLOnceWork() {
        delete mF;
    }
    virtual void readyToRun();
    virtual bool threadLoop();
    virtual void destroy();

private:
    std::function<void(void)>* mF;
};
#endif

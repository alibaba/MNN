//
//  GLWorkThread.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GLWorkThread.h"
namespace MNN {
GLWorkThread::GLWorkThread() : GLThread(false) {
    mContext = NULL;
}
GLWorkThread::~GLWorkThread() {
}

void GLWorkThread::onStop() {
    mWait4Work.post();
}

void GLWorkThread::onStart() {
}

std::shared_ptr<GLWorkSemore> GLWorkThread::queueWork(std::shared_ptr<GLWork> work, bool needSemore) {
    GLAutoLock _l(mWorkLock);
    std::shared_ptr<GLWorkSemore> s;
    if (needSemore) {
        s.reset(new Sema);
    }
    mWorks.push(std::make_pair(work, s));
    mWait4Work.post();
    return s;
}

bool GLWorkThread::threadLoop() {
    mWait4Work.wait();
    WORK w;
    GLAutoLock _l(mWorkLock);
    while (!mWorks.empty()) {
        w = mWorks.front();
        mWorks.pop();
        (w.first)->runOnePass();
        Sema* s = (Sema*)(w.second.get());
        if (nullptr != s) {
            s->post();
        }
    }
    return true;
}

void GLWorkThread::destroy() {
    GLContext::destroy(mContext);
}
void GLWorkThread::readyToRun() {
    mContext = GLContext::init();
}
} // namespace MNN

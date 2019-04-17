//
//  GLWorkThread.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLWORKTHREAD_H
#define GLWORKTHREAD_H
#include <queue>
#include "GLContext.h"
#include "GLThread.h"
#include "GLWork.h"
namespace MNN {
class GLWorkThread : public GLThread {
public:
    class Sema : public GLWorkSemore {
    public:
        virtual bool wait(int timeout_ms) {
            mSem.wait();
            return true;
        }
        void post() {
            mSem.post();
        }

    private:
        GLSema mSem;
    };
    std::shared_ptr<GLWorkSemore> queueWork(std::shared_ptr<GLWork> work, bool needSemore = true);
    GLWorkThread();
    virtual ~GLWorkThread();

protected:
    virtual void onStart();
    virtual void onStop();
    virtual void readyToRun();
    virtual bool threadLoop();
    virtual void destroy();

private:
    typedef std::pair<std::shared_ptr<GLWork>, std::shared_ptr<GLWorkSemore> > WORK;
    std::queue<WORK> mWorks;
    GLSema mWait4Work;
    GLLock mWorkLock;
    GLContext::nativeContext* mContext;
};
} // namespace MNN
#endif

//
//  GLLock.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLLOCK_H
#define GLLOCK_H
#include "core/Macro.h"
namespace MNN {
namespace OpenGL {
class GLLock {
public:
    GLLock();
    ~GLLock();
    void lock();
    void unlock();

private:
    void* mData;
};
class GLAutoLock {
public:
    GLAutoLock(GLLock& l) : mL(l) {
        mL.lock();
    }
    ~GLAutoLock() {
        mL.unlock();
    }

private:
    GLLock& mL;
};
} // namespace OpenGL
} // namespace MNN
#endif

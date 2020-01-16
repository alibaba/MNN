//
//  GLLock.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLLock.hpp"
#include <assert.h>
#include <pthread.h>
namespace MNN {
namespace OpenGL {
GLLock::GLLock() {
    pthread_mutex_t* m = new pthread_mutex_t;
    pthread_mutex_init(m, NULL);
    mData = (void*)m;
}

GLLock::~GLLock() {
    assert(NULL != mData);
    pthread_mutex_t* m = (pthread_mutex_t*)mData;
    pthread_mutex_destroy(m);
    delete m;
}

void GLLock::lock() {
    assert(NULL != mData);
    pthread_mutex_lock((pthread_mutex_t*)mData);
}

void GLLock::unlock() {
    assert(NULL != mData);
    pthread_mutex_unlock((pthread_mutex_t*)mData);
}
} // namespace OpenGL
} // namespace MNN

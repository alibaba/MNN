//
//  GLThread.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GLThread.h"
#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/stat.h>
namespace MNN {
void* GLThread::threadFunc(void* arg) {
    GLThread* t = (GLThread*)arg;
    t->run();
    return NULL;
}

GLThread::GLThread(bool _start) {
    pthread_t* d = new pthread_t;
    mData        = (void*)d;
    mRunning     = false;
}
GLThread::~GLThread() {
    stop();
    pthread_t* d = (pthread_t*)mData;
    delete d;
}

void GLThread::run() {
    this->readyToRun();
    bool rerun = true;
    do {
        rerun = this->threadLoop();
    } while (mRunning && rerun);
    this->destroy();
}

void GLThread::start() {
    GLAutoLock _l(mLock);
    if (!mRunning) {
        this->onStart();
        pthread_t* t = (pthread_t*)mData;
        pthread_create(t, NULL, GLThread::threadFunc, this);
        mRunning = true;
    }
}

void GLThread::stop() {
    GLAutoLock _l(mLock);
    if (mRunning) {
        mRunning = false;
        this->onStop();
        pthread_t* t = (pthread_t*)mData;
        pthread_join(*t, NULL);
    }
}

GLSema::GLSema() {
    sem_t* s   = new sem_t;
    auto error = sem_init(s, 0, 0);
    mData      = (void*)s;
    MNN_ASSERT(NULL != s);
}
GLSema::~GLSema() {
    MNN_ASSERT(NULL != mData);
    sem_destroy((sem_t*)mData);
    delete (sem_t*)mData;
}

void GLSema::wait() {
    MNN_ASSERT(NULL != mData);
    sem_t* s = (sem_t*)(mData);
    sem_wait(s);
}

void GLSema::post() {
    MNN_ASSERT(NULL != mData);
    sem_t* s = (sem_t*)(mData);
    sem_post(s);
}
} // namespace MNN

//
//  GLWork.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLWORK_H
#define GLWORK_H
#include <functional>
#include <memory>
#include "Macro.h"
namespace MNN {

class GLWork {
public:
    /*Run in GL Thread*/
    virtual bool onPrepare() = 0; // Create Shader, copy data to texture if needed
    virtual void onProcess() = 0; // Run (In most case, glDrawArray)
    virtual void onFinish()  = 0; // After this work is done, copy result if needed
    virtual void onDestroy() = 0;
    ; // destroy resource
    void runOnePass() {
        this->onPrepare();
        this->onProcess();
        this->onFinish();
        this->onDestroy();
    }
    /*Run in other Thread*/
    GLWork() {
    }
    virtual ~GLWork() {
    }
};

class GLFunctionWork : public GLWork {
public:
    GLFunctionWork(std::function<void()> f) : mFunc(f) {
    }
    virtual ~GLFunctionWork() {
    }
    /*Run in GL Thread*/
    virtual bool onPrepare() {
        return true;
    }
    virtual void onProcess() {
        mFunc();
    }
    virtual void onFinish() {
    }
    virtual void onDestroy() {
    }

private:
    std::function<void()> mFunc;
};

class GLWorkSemore {
public:
    GLWorkSemore() {
    }
    virtual ~GLWorkSemore() {
    }
    virtual bool wait(int timeout_ms = 0) = 0;
};
} // namespace MNN
#endif

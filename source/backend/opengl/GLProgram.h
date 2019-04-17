//
//  GLProgram.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLPROGRAM_H
#define GLPROGRAM_H

#include <string>
#include "GLHead.h"
#include "GLLock.h"
namespace MNN {
class GLProgram {
public:
    GLProgram(const std::string& computeShader);
    virtual ~GLProgram();

    inline unsigned int id() const {
        return mId;
    }

    static std::string getHead();

    /*These API must be called in openGL context Thread*/
    void use();
    int attr(const char* name) const;
    int uniform(const char* name) const;

protected:
    unsigned int mId = 0;

private:
    unsigned int mComputeId = 0;
};
} // namespace MNN

#endif

//
//  GLProgram.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLPROGRAM_H
#define GLPROGRAM_H

#include <string>
#include "backend/opengl/GLHead.hpp"
#include "backend/opengl/GLLock.hpp"
namespace MNN {
namespace OpenGL {
class GLProgram {
public:
    GLProgram(const std::string& computeShader);
    virtual ~GLProgram();

    inline unsigned int getProgramId() const {
        return mProgramId;
    }

    static std::string getHead(std::string imageFormat);

    /*These API must be called in openGL context Thread*/
    void useProgram();
    int getAttribLocation(const char* name) const;
    int getUniformLocation(const char* name) const;

private:
    bool compileShader(GLuint s);
    unsigned int mShaderId = 0;
    unsigned int mProgramId = 0;
};
} // namespace OpenGL
} // namespace MNN

#endif

//
//  GLSSBOBuffer.h
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef GLSSBOBUFFER_H
#define GLSSBOBUFFER_H

#include "backend/opengl/GLHead.hpp"
namespace MNN {
namespace OpenGL {
class GLSSBOBuffer {
public:
    GLSSBOBuffer(GLsizeiptr size, GLenum type = GL_SHADER_STORAGE_BUFFER, GLenum usage = GL_DYNAMIC_DRAW);
    ~GLSSBOBuffer();

    GLuint getId() const {
        return mId;
    }
    void* map(GLbitfield bufMask);
    void unmap();

    GLsizeiptr size() const {
        return mSize;
    }

private:
    GLuint mId = 0;
    GLsizeiptr mSize;
    GLenum mType;
};
} // namespace OpenGL
} // namespace MNN
#endif // GLSSBOBUFFER_H

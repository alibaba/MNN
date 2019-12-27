//
//  GLSSBOBuffer.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLSSBOBuffer.hpp"
namespace MNN {
namespace OpenGL {
GLSSBOBuffer::GLSSBOBuffer(GLsizeiptr size, GLenum type, GLenum usage) {
    mType = type;
    GLASSERT(size > 0);
    glGenBuffers(1, &mId);
    OPENGL_CHECK_ERROR;
    glBindBuffer(mType, mId);
    OPENGL_CHECK_ERROR;
    GLASSERT(mId > 0);
    glBufferData(mType, size, NULL, usage);
    OPENGL_CHECK_ERROR;
    mSize = size;
}

GLSSBOBuffer::~GLSSBOBuffer() {
    glDeleteBuffers(1, &mId);
    OPENGL_CHECK_ERROR;
}

void *GLSSBOBuffer::map(GLbitfield bufMask) {
    glBindBuffer(mType, mId);
    OPENGL_CHECK_ERROR;
    auto ptr = glMapBufferRange(mType, 0, mSize, bufMask);
    OPENGL_CHECK_ERROR;
    return ptr;
}

void GLSSBOBuffer::unmap() {
    glBindBuffer(mType, mId);
    glUnmapBuffer(mType);
    OPENGL_CHECK_ERROR;
}
} // namespace OpenGL
} // namespace MNN

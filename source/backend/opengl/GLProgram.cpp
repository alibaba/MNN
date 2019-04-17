//
//  GLProgram.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "GLProgram.h"
#include <string.h>
#include <fstream>
#include <sstream>
#include "GLDebug.h"
using namespace std;

namespace MNN {
GLProgram::~GLProgram() {
    glDeleteProgram(mId);
    glDeleteShader(mComputeId);
    OPENGL_CHECK_ERROR;
}
static bool compileShader(GLuint s) {
    GLint status;
    glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &status);
    if (!status) {
        int len;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        if (0 >= len) {
            glGetShaderInfoLog(s, 0, &len, NULL);
        }
        char* buffer = new char[len + 1];
        glGetShaderInfoLog(s, len, NULL, buffer);
        buffer[len] = 0;
        FUNC_PRINT_ALL(buffer, s);
        delete[] buffer;
        return false;
    }
    return true;
}

int GLProgram::attr(const char* name) const {
    GLASSERT(NULL != name && 0 != mId);
    return glGetAttribLocation(mId, name);
}
int GLProgram::uniform(const char* name) const {
    GLASSERT(NULL != name && 0 != mId);
    return glGetUniformLocation(mId, name);
}

void GLProgram::use() {
    glUseProgram(mId);
    OPENGL_CHECK_ERROR;
}

GLProgram::GLProgram(const std::string& computeShader) {
    /*Create Shader*/
    mComputeId = glCreateShader(GL_COMPUTE_SHADER);
    OPENGL_CHECK_ERROR;
    const char* _ver[1];
    _ver[0] = computeShader.c_str();
    glShaderSource(mComputeId, 1, _ver, NULL);
    OPENGL_CHECK_ERROR;
    /*TODO move GLASSERT to be log*/
    bool res = compileShader(mComputeId);
    // if (!res) FUNC_PRINT_ALL(mVertex.c_str(), s);
    GLASSERT(res);
    /*Create Program*/
    mId = glCreateProgram();
    OPENGL_CHECK_ERROR;
    glAttachShader(mId, mComputeId);
    OPENGL_CHECK_ERROR;
    glLinkProgram(mId);
    OPENGL_CHECK_ERROR;
    GLint linked;
    glGetProgramiv(mId, GL_LINK_STATUS, &linked);
    if (!linked) {
        FUNC_PRINT(linked);
        GLsizei len;
        glGetProgramiv(mId, GL_INFO_LOG_LENGTH, &len);
        if (len <= 0) {
            glGetProgramInfoLog(mId, 0, &len, NULL);
        }
        if (len > 0) {
            char* buffer = new char[len + 1];
            buffer[len]  = '\0';
            glGetProgramInfoLog(mId, len, NULL, buffer);
            FUNC_PRINT_ALL(buffer, s);
            delete[] buffer;
        }
    }
}

std::string GLProgram::getHead() {
    std::ostringstream headOs;
    headOs << "#version 310 es\n";
    headOs << "#define PRECISION mediump\n";
    headOs << "precision PRECISION float;\n";
    headOs << "#define FORMAT " << IMAGE_FORMAT << "\n";
    return headOs.str();
}
} // namespace MNN

//
//  GLProgram.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opengl/GLProgram.hpp"
#include <string.h>
#include <fstream>
#include <sstream>
#include "backend/opengl/GLDebug.hpp"
using namespace std;

namespace MNN {
namespace OpenGL {
GLProgram::~GLProgram() {
    glDeleteShader(mShaderId);
    glDeleteProgram(mProgramId);
    OPENGL_CHECK_ERROR;
}
bool GLProgram::compileShader(GLuint s) {
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

int GLProgram::getAttribLocation(const char* name) const {
    GLASSERT(NULL != name && 0 != mProgramId);
    return glGetAttribLocation(mProgramId, name);
}
int GLProgram::getUniformLocation(const char* name) const {
    GLASSERT(NULL != name && 0 != mProgramId);
    return glGetUniformLocation(mProgramId, name);
}

void GLProgram::useProgram() {
    glUseProgram(mProgramId);
    OPENGL_CHECK_ERROR;
}

GLProgram::GLProgram(const std::string& computeShader) {
    /*Create Shader*/
    mShaderId = glCreateShader(GL_COMPUTE_SHADER);
    OPENGL_CHECK_ERROR;
    const char* _ver[1];
    _ver[0] = computeShader.c_str();
    glShaderSource(mShaderId, 1, _ver, NULL);
    OPENGL_CHECK_ERROR;
    /*TODO move GLASSERT to be log*/
    bool res = compileShader(mShaderId);
    // if (!res) FUNC_PRINT_ALL(mVertex.c_str(), s);
    GLASSERT(res);
    /*Create Program*/
    mProgramId = glCreateProgram();
    OPENGL_CHECK_ERROR;
    glAttachShader(mProgramId, mShaderId);
    OPENGL_CHECK_ERROR;
    glLinkProgram(mProgramId);
    OPENGL_CHECK_ERROR;
    GLint linked;
    glGetProgramiv(mProgramId, GL_LINK_STATUS, &linked);
    if (!linked) {
//        FUNC_PRINT(linked);
        GLsizei len;
        glGetProgramiv(mProgramId, GL_INFO_LOG_LENGTH , &len);
        if (len <= 0) {
            glGetProgramInfoLog(mProgramId, 0, &len, NULL);
        }
        if (len > 0) {
            char* buffer = new char[len + 1];
            buffer[len]  = '\0';
            glGetProgramInfoLog(mProgramId, len, NULL, buffer);
            FUNC_PRINT_ALL(buffer, s);
            delete[] buffer;
        }
    }
}

std::string GLProgram::getHead(std::string imageFormat) {
    std::ostringstream headOs;
    headOs << "#version 310 es\n";
    headOs << "#define PRECISION mediump\n";
    headOs << "precision PRECISION float;\n";
    headOs << "#define FORMAT " << imageFormat << "\n";
    return headOs.str();
}
} // namespace OpenGL
} // namespace MNN

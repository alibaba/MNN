//
//  GLUtils.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "backend/opengl/GLUtils.hpp"
#include "backend/opengl/GLBackend.hpp"
#include <sstream>
namespace MNN {
namespace OpenGL {
    void setLocalSize(std::vector<std::string>& prefix, int* localSize, int setLocalSizeX, int setLocalSizeY, int setLocalSizeZ){

        GLint maxLocalSizeX, maxLocalSizeY, maxLocalSizeZ;
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxLocalSizeX);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxLocalSizeY);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &maxLocalSizeZ);
        localSize[0]     = setLocalSizeX < maxLocalSizeX ? setLocalSizeX : maxLocalSizeX;
        localSize[1]     = setLocalSizeY < maxLocalSizeY ? setLocalSizeY : maxLocalSizeY;
        localSize[2]     = setLocalSizeZ < maxLocalSizeZ ? setLocalSizeZ : maxLocalSizeZ;

        {
            std::ostringstream os;
            os << "#define XLOCAL " << localSize[0];
            prefix.push_back(os.str());
        }
        {
            std::ostringstream os;
            os << "#define YLOCAL " << localSize[1];
            prefix.push_back(os.str());
        }
        {
            std::ostringstream os;
            os << "#define ZLOCAL " << localSize[2];
            prefix.push_back(os.str());
        }
    }
} // namespace OpenGL
} // namespace MNN

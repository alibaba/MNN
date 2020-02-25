//
//  winogradGenerateGLSL.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include <fstream>
#include <sstream>
#include <MNN/MNNDefine.h>
#include "math/Matrix.hpp"
#include "math/WingoradGenerater.hpp"

using namespace std;

const char* gWinogradSourceHead =
    "#version 450 core\n"
    "layout(std430) buffer;\n"
    "layout(std430) uniform;\n"
    "layout(set=0, binding=0, rgba16f) writeonly restrict uniform image2D uOutput;\n"
    "layout(set=0, binding=1) uniform sampler3D uInput;\n"
    "layout(set=0, binding=2) readonly restrict uniform constBuffer {\n"
    "    ivec4 inputSize;\n"
    "    ivec4 outputSize;\n"
    "    int padX;\n"
    "    int padY;\n"
    "    int unitWidth;\n"
    "    int unitHeight;\n"
    "    int unit;\n"
    "} uConst;\n"
    "layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;\n"
    "void main()\n"
    "{\n"
    "    ivec3 pos = ivec3(gl_GlobalInvocationID);\n"
    "    if (pos.x < uConst.unitWidth && pos.y < uConst.unitHeight)\n"
    "    {\n"
    "int dstXOrigin = pos.z;\n"
    "int dstYOrigin = uConst.unitWidth * pos.y + pos.x;\n"
    "int srcHeight = (uConst.unitWidth*uConst.unitHeight+3)/4;\n"
    "int dstY = dstYOrigin / 4;\n"
    "int dstX = dstYOrigin % 4 + 4*dstXOrigin;\n"
    "        int sxStart = pos.x*uConst.unit - uConst.padX;\n"
    "        int syStart = pos.y*uConst.unit - uConst.padY;\n";

const char* gWinogradSourceTail =
    "    }\n"
    "}\n";

const char* gWinogradDestHead =
    "#version 450 core\n"
    "layout(std430) buffer;\n"
    "layout(std430) uniform;\n"
    "layout(set=0, binding=0, rgba16f) writeonly restrict uniform image3D uOutput;\n"
    "layout(set=0, binding=1) uniform sampler2D uInput;\n"
    "layout(set=0, binding=2) uniform sampler2D uBias;\n"
    "layout(set=0, binding=3) readonly restrict uniform constBuffer {\n"
    "    ivec4 inputSize;\n"
    "    ivec4 outputSize;\n"
    "    int padX;\n"
    "    int padY;\n"
    "    int unitWidth;\n"
    "    int unitHeight;\n"
    "    int unit;\n"
    "} uConst;\n"
    "layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;\n"
    "void main()\n"
    "{\n"
    "    ivec3 pos = ivec3(gl_GlobalInvocationID);\n"

    "    if (pos.x < uConst.unitWidth && pos.y < uConst.unitHeight)\n"
    "    {\n"
    "int dstWidth = (uConst.unitWidth*uConst.unitHeight+3)/4;\n"
    "int dstXOrigin = uConst.unitWidth * pos.y + pos.x;\n"
    "int dstX = dstXOrigin / 4;\n"
    "int dstY = 4*pos.z + dstXOrigin % 4;\n"
    "        vec4 bias = texelFetch(uBias, ivec2(pos.z, 0), 0);\n"
    "int oyStart = pos.y * uConst.unit;\n"
    "int oxStart = pos.x * uConst.unit;\n"
    "int oz = pos.z;\n";

const char* gWinogradDestTail =
    "    }\n"
    "}\n";

int main(int argc, const char* argv[]) {
    int unit       = atoi(argv[1]);
    int kernelSize = atoi(argv[2]);
    auto alpha     = unit + kernelSize - 1;
    float interp   = 0.5f;
    if (argc > 3) {
        interp = atof(argv[3]);
    }
    MNN::Math::WinogradGenerater generater(unit, kernelSize, interp);
    auto a = generater.A();
    auto b = generater.B();
    auto g = generater.G();

    MNN::Math::Matrix::print(a.get(), "A");
    MNN::Math::Matrix::print(b.get(), "B");
    MNN::Math::Matrix::print(g.get(), "G");
    std::ostringstream sourceFileOstream;
    { sourceFileOstream << "winogradTransformSource" << unit << "_" << kernelSize << "_" << interp << ".comp"; }
    auto sourceFile = sourceFileOstream.str();
    MNN_PRINT("Generate %s\n", sourceFile.c_str());
    {
        std::ofstream sourceOutput(sourceFile.c_str());
        sourceOutput << gWinogradSourceHead << "{\n";

        // Load
        for (int y = 0; y < alpha; ++y) {
            for (int x = 0; x < alpha; ++x) {
                sourceOutput << "vec4 S" << x << y << "= texelFetch(uInput, ivec3(sxStart+" << x << ", syStart+ " << y
                             << ", pos.z), 0);\n";
            }
        }

        // M = BT*S
        auto bFloat = b->host<float>();
        for (int y = 0; y < alpha; ++y) {
            for (int x = 0; x < alpha; ++x) {
                sourceOutput << "vec4 m" << x << y << "= ";

                for (int k = 0; k < alpha; ++k) {
                    auto value = bFloat[alpha * k + y];
                    if (0.0f == value) {
                        continue;
                    } else if (1.0f == value) {
                        sourceOutput << "+S" << x << k;
                    } else if (-1.0f == value) {
                        sourceOutput << "-S" << x << k;
                    } else {
                        if (value > 0) {
                            sourceOutput << "+";
                        }
                        sourceOutput << value << "*S" << x << k;
                    }
                }
                sourceOutput << ";\n";
            }
        }

        // S = M*B
        for (int y = 0; y < alpha; ++y) {
            for (int x = 0; x < alpha; ++x) {
                sourceOutput << "imageStore(uOutput, ivec2(dstX, dstY+srcHeight*" << (y * alpha + x) << "), ";
                for (int k = 0; k < alpha; ++k) {
                    auto value = bFloat[alpha * k + x];
                    if (0.0f == value) {
                        continue;
                    } else if (1.0f == value) {
                        sourceOutput << "+m" << k << y;
                    } else if (-1.0f == value) {
                        sourceOutput << "-m" << k << y;
                    } else {
                        if (value > 0) {
                            sourceOutput << "+";
                        }
                        sourceOutput << value << "*m" << k << y;
                    }
                }
                sourceOutput << ");\n";
            }
        }

        sourceOutput << "}\n";
        sourceOutput << gWinogradSourceTail;
    }

    std::ostringstream destFileOstream;
    { destFileOstream << "winogradTransformDest" << unit << "_" << kernelSize << "_" << interp << ".comp"; }
    auto destFile = destFileOstream.str();
    MNN_PRINT("Generate %s\n", destFile.c_str());
    {
        std::ofstream destFileOs(destFile.c_str());
        destFileOs << gWinogradDestHead;
        destFileOs << "{\n";
        auto aFloat = a->host<float>();

        // Load
        for (int y = 0; y < alpha; ++y) {
            for (int x = 0; x < alpha; ++x) {
                destFileOs << "vec4 S" << x << y << "= texelFetch(uInput, ivec2(dstX+dstWidth*" << (x + y * alpha)
                           << ", dstY), 0);\n";
            }
        }

        // M = AT* S
        for (int y = 0; y < unit; ++y) {
            for (int x = 0; x < alpha; ++x) {
                destFileOs << "vec4 m" << x << y << "= ";
                for (int k = 0; k < alpha; ++k) {
                    auto value = aFloat[k * unit + y];
                    if (0.0f == value) {
                        continue;
                    } else if (1.0f == value) {
                        destFileOs << "+S" << x << k;
                    } else if (-1.0f == value) {
                        destFileOs << "-S" << x << k;
                    } else {
                        if (value > 0) {
                            destFileOs << "+";
                        }
                        destFileOs << value << "*S" << x << k;
                    }
                }
                destFileOs << ";\n";
            }
        }

        // S = M * A
        for (int y = 0; y < unit; ++y) {
            for (int x = 0; x < unit; ++x) {
                destFileOs << "{\n";
                destFileOs << "vec4 res = bias";
                for (int k = 0; k < alpha; ++k) {
                    auto value = aFloat[k * unit + x];
                    if (0.0f == value) {
                        continue;
                    } else if (1.0f == value) {
                        destFileOs << "+m" << k << y;
                    } else if (-1.0f == value) {
                        destFileOs << "-m" << k << y;
                    } else {
                        if (value > 0) {
                            destFileOs << "+";
                        }
                        destFileOs << value << "*m" << k << y;
                    }
                }
                destFileOs << ";\n";

                destFileOs << "#ifdef RELU\n";
                destFileOs << "res = max(res, vec4(0));\n";
                destFileOs << "#endif\n";
                destFileOs << "#ifdef RELU6\n";
                destFileOs << "res = clamp(res, vec4(0), vec4(6));\n";
                destFileOs << "#endif\n";
                destFileOs << "imageStore(uOutput, ivec3(oxStart+" << x << ", oyStart+" << y << ", pos.z), res);\n";

                destFileOs << "}\n";
            }
        }

        destFileOs << "}\n";
        destFileOs << gWinogradDestTail;
    }

    return 0;
}

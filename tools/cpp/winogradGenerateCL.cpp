//
//  winogradGenerateCL.cpp
//  MNN
//
//  Created by MNN on 2019/01/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include <string.h>
#include <fstream>
#include <sstream>
#include <MNN/MNNDefine.h>
#include "math/Matrix.hpp"
#include "math/WingoradGenerater.hpp"

using namespace std;

const char* gWinogradSourceHead =
    "#include <common.h>\n"

    "__kernel void winogradTransformSource\n"
    "(\n"
    "__read_only image2d_t uInput, //0\n"
    "__write_only image2d_t uOutput,\n"
    "__private const int unitWidth,\n"
    "__private const int unitHeight,//3\n"
    "__private const int padX,\n"
    "__private const int padY,\n"
    "__private const int srcWidth,//6\n"
    "__private const int srcHeight,\n"
    "__private const int srcChannelC4,\n"
    "__private const int offsetX,//9\n"
    "__private const int offsetY\n"
    ")\n"
    "{\n"
    "int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));\n"
    "if (pos.x < unitWidth && pos.y < unitHeight)\n"
    "{\n"
    "int2 realPos = (int2)(pos.x+offsetX, pos.y+offsetY);\n"
    "int dstXOrigin = pos.z;\n"
    "int batchIndex = pos.z / srcChannelC4;\n"
    "int srcZ = pos.z % srcChannelC4;\n"
    "int dstYOrigin = unitWidth * pos.y + pos.x;\n"
    "int dstHeight = (unitWidth*unitHeight+3)/4;\n"
    "int dstY = dstYOrigin / 4;\n"
    "int dstX = dstYOrigin % 4 + 4*dstXOrigin;\n";

const char* gWinogradSourceTail =
    "    }\n"
    "}\n";

const char* gWinogradDestHead =
    "#include <common.h>\n"
    "__kernel void winogradTransformDest\n"
    "(\n"
    "__read_only image2d_t uInput, //0\n"
    "__read_only image2d_t uBias,\n"
    "__write_only image2d_t uOutput,\n"
    "__private const int unitWidth,//3\n"
    "__private const int unitHeight,\n"
    "__private const int dstWidth,\n"
    "__private const int dstHeight,//6\n"
    "__private const int dstChannelC4,\n"
    "__private const int offsetX,\n"
    "__private const int offsetY\n"
    ")\n"
    "{\n"
    "int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));\n"
    "if (pos.x < unitWidth && pos.y < unitHeight)\n"
    "{\n"
    "int2 realPos = (int2)(pos.x+offsetX, pos.y+offsetY);\n"
    "int srcWidth = (unitWidth*unitHeight+3)/4;\n"
    "int dstXOrigin = unitWidth * pos.y + pos.x;\n"
    "int dstX = dstXOrigin / 4;\n"
    "int dstY = 4*pos.z + dstXOrigin % 4;\n"
    "int oz = pos.z % dstChannelC4;\n"
    "DATA_TYPE4 bias = READ_IMAGE(uBias, SAMPLER, (int2)(oz, 0));\n"
    "int batchIndex = pos.z / dstChannelC4;\n";

const char* gWinogradDestTail =
    "    }\n"
    "}\n";
inline void _printFloat(ostream& output, float v) {
    if (v == floor(v)) {
        output << v << ".0h";
    } else {
        output << v << "h";
    }
}

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
    { sourceFileOstream << "winogradTransformSource" << unit << "_" << kernelSize << "_" << interp << ".cl"; }
    auto sourceFile = sourceFileOstream.str();
    MNN_PRINT("Generate %s\n", sourceFile.c_str());
    {
        std::ofstream sourceOutput(sourceFile.c_str());
        sourceOutput << gWinogradSourceHead << "{\n";
        sourceOutput << "int sxStart = (realPos.x)*" << unit << " - padX;\n";
        sourceOutput << "int syStart = (realPos.y)*" << unit << " - padY;\n";

        // Load
        for (int y = 0; y < alpha; ++y) {
            for (int x = 0; x < alpha; ++x) {
                sourceOutput << "DATA_TYPE4 S" << x << y << ";\n";
            }
        }
        for (int y = 0; y < alpha; ++y) {
            for (int x = 0; x < alpha; ++x) {
                sourceOutput << "{\n";
                sourceOutput << "int sx = " << x << "+sxStart;\n";
                sourceOutput << "int sy = " << y << "+syStart;\n";
                sourceOutput << "int imageSx = select(sx + srcZ*srcWidth, -1, sx<0 || sx >= srcWidth);\n";
                sourceOutput << "int imageSy = select(batchIndex*srcHeight + sy, -1, sy<0 || sy >= srcHeight);\n";

                sourceOutput << "S" << x << y << "= READ_IMAGE(uInput, SAMPLER, (int2)(imageSx, imageSy));\n";
                sourceOutput << "}\n";
            }
        }

        // M = BT*S
        auto bFloat = b->host<float>();
        for (int y = 0; y < alpha; ++y) {
            for (int x = 0; x < alpha; ++x) {
                sourceOutput << "DATA_TYPE4 m" << x << y << "= ";

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
                        _printFloat(sourceOutput, value);
                        sourceOutput << "*S" << x << k;
                    }
                }
                sourceOutput << ";\n";
            }
        }

        // S = M*B
        for (int y = 0; y < alpha; ++y) {
            for (int x = 0; x < alpha; ++x) {
                sourceOutput << "WRITE_IMAGE(uOutput, (int2)(dstX, dstY+dstHeight*" << (y * alpha + x) << "), ";
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
                        _printFloat(sourceOutput, value);

                        sourceOutput << "*m" << k << y;
                    }
                }
                sourceOutput << ");\n";
            }
        }

        sourceOutput << "}\n";
        sourceOutput << gWinogradSourceTail;
    }

    std::ostringstream destFileOstream;
    { destFileOstream << "winogradTransformDest" << unit << "_" << kernelSize << "_" << interp << ".cl"; }
    auto destFile = destFileOstream.str();
    MNN_PRINT("Generate %s\n", destFile.c_str());
    {
        std::ofstream destFileOs(destFile.c_str());
        destFileOs << gWinogradDestHead;
        destFileOs << "{\n";
        auto aFloat = a->host<float>();
        destFileOs << " int oyStart = realPos.y * " << unit << ";\n";
        destFileOs << " int oxStart = realPos.x * " << unit << ";\n";
        // Load
        for (int y = 0; y < alpha; ++y) {
            for (int x = 0; x < alpha; ++x) {
                destFileOs << "DATA_TYPE4 S" << x << y << "= READ_IMAGE(uInput, SAMPLER, (int2)(dstX+srcWidth*"
                           << (x + y * alpha) << ", dstY));\n";
            }
        }

        // M = AT* S
        for (int y = 0; y < unit; ++y) {
            for (int x = 0; x < alpha; ++x) {
                destFileOs << "DATA_TYPE4 m" << x << y << "= ";
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
                        _printFloat(destFileOs, value);
                        destFileOs << "*S" << x << k;
                    }
                }
                destFileOs << ";\n";
            }
        }

        // S = M * A
        for (int y = 0; y < unit; ++y) {
            for (int x = 0; x < unit; ++x) {
                destFileOs << "{\n";
                destFileOs << "int ox = oxStart + " << x << ";\n";
                destFileOs << "int oy = oyStart + " << y << ";\n";
                destFileOs << "if (ox < dstWidth && oy < dstHeight)\n{\n";
                destFileOs << "int imageOx = ox+oz*dstWidth;\n";
                destFileOs << "int imageOy = oy+batchIndex*dstHeight;\n";
                destFileOs << "DATA_TYPE4 res = bias";
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
                        _printFloat(destFileOs, value);
                        destFileOs << "*m" << k << y;
                    }
                }
                destFileOs << ";\n";

                destFileOs << "#ifdef RELU\n";
                destFileOs << "res = max(res, (DATA_TYPE4)(0));\n";
                destFileOs << "#endif\n";
                destFileOs << "#ifdef RELU6\n";
                destFileOs << "res = clamp(res, (DATA_TYPE4)(0), (DATA_TYPE4)(6));\n";
                destFileOs << "#endif\n";
                destFileOs << "WRITE_IMAGE(uOutput, (int2)(imageOx, imageOy), res);\n";
                destFileOs << "}\n";

                destFileOs << "}\n";
            }
        }

        destFileOs << "}\n";
        destFileOs << gWinogradDestTail;
    }

    return 0;
}

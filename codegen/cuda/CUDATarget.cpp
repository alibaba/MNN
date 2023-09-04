//
//  CUDATarget.cpp
//  MNN
//
//  Created by MNN on 2023/06/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "core/TensorUtils.hpp"
#include "MNN_generated.h"
#include "CUDATarget.hpp"
#include "core/OpCommonUtils.hpp"

namespace MNN {
std::string CUDATarget::type() {
    if(mVectorize) {
        if(mPrecision == BackendConfig::Precision_Low) {
            return "half2 ";
        }
        return "float4 ";
    }
    if(mPrecision == BackendConfig::Precision_Low) {
        return "half ";
    }
    return "float ";
}
std::string CUDATarget::macro() {
    std::string headStr;
    if(mPrecision == BackendConfig::Precision_Low) {
        headStr += "#include <cuda_fp16.h>\n";
    }
    // headStr += "#include \"MNNCUDAFunction.cuh\"\n";
    headStr += "#define OFFSET_CHECK\\\n"
        "\tsize_t offset = blockIdx.x * blockDim.x + threadIdx.x;\\\n"
        "\tif (offset >= count) { return; }\n";
    return headStr;
}
std::string CUDATarget::number(float val) {
    std::string tmpStr = std::to_string(val);
    if(mVectorize) {
        if(mPrecision == BackendConfig::Precision_Low) {
            return "make_half2((half)" + tmpStr + ",(half)" + tmpStr + ")";
        }
        return "make_float4(" + tmpStr + "," + tmpStr + "," + tmpStr + "," + tmpStr + ")";
    }
    return tmpStr;
}
std::string CUDATarget::codegen(std::vector<std::string>& inputs, const Command* cmd, std::string& inpName) {
    const Op* op = cmd->op;
    std::stringstream ss; 
    switch (op->type()) {
        case MNN::OpType_BinaryOp:
        {
            auto lhs = inputs[0], rhs = inputs[1];
            auto type = static_cast<MNN::BinaryOpOperation>(op->main_as_BinaryOp()->opType());
            switch (type) {
                case BinaryOpOperation_ADD:
                    if(mVectorize) {
                        ss << inpName << ".x=(" << lhs << ".x+" << rhs << ".x);\n";
                        ss << inpName << ".y=(" << lhs << ".y+" << rhs << ".y)";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";
                            ss << inpName << ".z=(" << lhs << ".z+" << rhs << ".z);\n";
                            ss << inpName << ".w=(" << lhs << ".w+" << rhs << ".w)";
                        }
                    } else {
                        ss << inpName << "=(" << lhs << "+" << rhs << ")";
                    }
                    break;
                case BinaryOpOperation_SUB:
                    if(mVectorize) {
                        ss << inpName << ".x=(" << lhs << ".x-" << rhs << ".x);\n";
                        ss << inpName << ".y=(" << lhs << ".y-" << rhs << ".y)";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";
                            ss << inpName << ".z=(" << lhs << ".z-" << rhs << ".z);\n";
                            ss << inpName << ".w=(" << lhs << ".w-" << rhs << ".w)";
                        }
                    } else {
                        ss << inpName << "=(" << lhs << "-" << rhs << ")";
                    }
                    break;
                case BinaryOpOperation_MUL:
                    if(mVectorize) {
                        ss << inpName << ".x=(" << lhs << ".x*" << rhs << ".x);\n";
                        ss << inpName << ".y=(" << lhs << ".y*" << rhs << ".y)";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";
                            ss << inpName << ".z=(" << lhs << ".z*" << rhs << ".z);\n";
                            ss << inpName << ".w=(" << lhs << ".w*" << rhs << ".w)";
                        }
                    } else {
                        ss << inpName << "=(" << lhs << "*" << rhs << ")";
                    }
                    break;
                case BinaryOpOperation_POW:
                    if(mVectorize) {
                        ss << inpName << ".x=pow((float)" << lhs << ".x,(float)" << rhs << ".x);\n";
                        ss << inpName << ".y=pow((float)" << lhs << ".y,(float)" << rhs << ".y)";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                        
                            ss << inpName << ".z=pow((float)" << lhs << ".z,(float)" << rhs << ".z);\n";
                            ss << inpName << ".w=pow((float)" << lhs << ".w,(float)" << rhs << ".w)";
                        }
                    } else {
                        ss << inpName << "=(pow((float)" << lhs << ",(float)" << rhs << "))";
                    }
                    break;
                case BinaryOpOperation_DIV:
                    if(mVectorize) {
                        ss << inpName << ".x=(" << lhs << ".x/" << rhs << ".x);\n";
                        ss << inpName << ".y=(" << lhs << ".y/" << rhs << ".y)";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(" << lhs << ".z/" << rhs << ".z);\n";
                            ss << inpName << ".w=(" << lhs << ".w/" << rhs << ".w)";
                        }
                    } else {
                        ss << inpName << "=(" << lhs << "/" << rhs << ")";
                    }
                    break;
                case BinaryOpOperation_MAXIMUM:
                    if(mVectorize) {
                        ss << inpName << ".x=fmax(" << lhs << ".x," << rhs << ".x);\n";
                        ss << inpName << ".y=fmax(" << lhs << ".y," << rhs << ".y)";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=fmax(" << lhs << ".z," << rhs << ".z);\n";
                            ss << inpName << ".w=fmax(" << lhs << ".w," << rhs << ".w)";
                        }
                    } else {
                        ss << inpName << "=(fmax(" << lhs << "," << rhs << "))";
                    }
                    break;
                case BinaryOpOperation_MINIMUM:
                    if(mVectorize) {
                        ss << inpName << ".x=fmin(" << lhs << ".x," << rhs << ".x);\n";
                        ss << inpName << ".y=fmin(" << lhs << ".y," << rhs << ".y)";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                          
                            ss << inpName << ".z=fmin(" << lhs << ".z," << rhs << ".z);\n";
                            ss << inpName << ".w=fmin(" << lhs << ".w," << rhs << ".w)";
                        }
                    } else {
                        ss << inpName << "=(fmin(" << lhs << "," << rhs << "))";
                    }
                    break;
                case BinaryOpOperation_REALDIV:
                    if(mVectorize) {
                        ss << inpName << ".x=(" << lhs << ".x/" << rhs << ".x);\n";
                        ss << inpName << ".y=(" << lhs << ".y/" << rhs << ".y)";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(" << lhs << ".z/" << rhs << ".z);\n";
                            ss << inpName << ".w=(" << lhs << ".w/" << rhs << ".w)";
                        }
                    } else {
                        ss << inpName << "=(" << lhs << "/" << rhs << ")";//ss << "((" << rhs << ") > 0.0 ? 1.0 : ((" << rhs << ") < 0.0 ? -1.0 : 0.0) * " << lhs << "/ max(abs(" << rhs << "), 0.0000001))";
                    }
                    break;
                default:
                    MNN_PRINT("Error: CUDA CodeGen not support Binary type:%d\n", type);
                    break;
            }
            break;
        }
        case MNN::OpType_Eltwise:
        {
            auto type = op->main_as_Eltwise()->type();
            switch (type) {
                case EltwiseType_SUM:
                case EltwiseType_SUB:
                case EltwiseType_PROD:
                {
                    std::unordered_map<int, std::string> elemToOp = {
                        {EltwiseType_PROD, "*"}, {EltwiseType_SUM, "+"}, {EltwiseType_SUB, "-"}
                    };
                    if(mVectorize) {
                        ss << inpName << ".x=(" << inputs[0] << ".x" << elemToOp[type] << inputs[1] << ".x";
                        for (int i = 2; i < inputs.size(); i++) {
                            ss << elemToOp[type] << inputs[i] << ".x";
                        }
                        ss << ");\n";

                        ss << inpName << ".y=(" << inputs[0] << ".y" << elemToOp[type] << inputs[1] << ".y";
                        for (int i = 2; i < inputs.size(); i++) {
                            ss << elemToOp[type] << inputs[i] << ".y";
                        }
                        ss << ")";

                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";   
                            ss << inpName << ".z=(" << inputs[0] << ".z" << elemToOp[type] << inputs[1] << ".z";
                            for (int i = 2; i < inputs.size(); i++) {
                                ss << elemToOp[type] << inputs[i] << ".z";
                            }
                            ss << ");\n";

                            ss << inpName << ".w=(" << inputs[0] << ".w" << elemToOp[type] << inputs[1] << ".w";
                            for (int i = 2; i < inputs.size(); i++) {
                                ss << elemToOp[type] << inputs[i] << ".w";
                            }
                            ss << ")";
                        }
                    } else {
                        ss << inpName << "=(" << inputs[0] << elemToOp[type] << inputs[1];
                        for (int i = 2; i < inputs.size(); i++) {
                            ss << elemToOp[type] << inputs[i];
                        }
                        ss << ")";
                    }
                    break;
                }
                case EltwiseType_MAXIMUM:
                {
                    if(mVectorize) {
                        MNN_PRINT("Error: CUDA CodeGen not support Eltwise  Parallel type:%d, Please Fix it\n", type);
                    }
                    std::function<std::string(int)> fmax = [&inputs, &fmax](int d) {
                        if (d == inputs.size() - 1) {
                            return inputs[d];
                        }
                        return "fmax(" + inputs[d] + ", " + fmax(d+1) + ")";
                    };
                    ss << inpName << "=" << fmax(0);
                    break;
                }
                default:
                    MNN_PRINT("Error: CUDA CodeGen not support Eltwise type:%d\n", type);
                    break;
            }
            break;
        }
        case MNN::OpType_UnaryOp:
        {
            auto unary = op->main_as_UnaryOp();
            auto type = unary->opType();
            auto operand = inputs[0];
            switch (type) {
                case UnaryOpOperation_SQUARE:
                    if(mVectorize) {
                        ss << inpName << ".x=(" << operand << ".x * " << operand << ".x);\n";
                        ss << inpName << ".y=(" << operand << ".y * " << operand << ".y)";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";   
                            ss << inpName << ".z=(" << operand << ".z * " << operand << ".z);\n";
                            ss << inpName << ".w=(" << operand << ".w * " << operand << ".w)";
                        }
                    } else {
                        ss << inpName << "=(" << operand << " * " << operand << ")";
                    }
                    break;
                case UnaryOpOperation_ERF:
                    if(mVectorize) {
                        ss << inpName << ".x=(erf(" << operand << ".x));\n";
                        ss << inpName << ".y=(erf(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(erf(" << operand << ".z));\n";
                            ss << inpName << ".w=(erf(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(erf(" << operand << "))";
                    }
                    break;
                case UnaryOpOperation_ERFC:
                    if(mVectorize) {
                        ss << inpName << ".x=(erfc(" << operand << ".x));\n";
                        ss << inpName << ".y=(erfc(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(erfc(" << operand << ".z));\n";
                            ss << inpName << ".w=(erfc(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(erfc(" << operand << "))";
                    }
                    break;
                case UnaryOpOperation_ERFINV:
                    if(mVectorize) {
                        ss << inpName << ".x=(erfinv(" << operand << ".x));\n";
                        ss << inpName << ".y=(erfinv(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(erfinv(" << operand << ".z));\n";
                            ss << inpName << ".w=(erfinv(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(erfinv(" << operand << "))";
                    }
                    break;
                case UnaryOpOperation_SQRT:
                    if(mVectorize) {
                        ss << inpName << ".x=(sqrt(" << operand << ".x));\n";
                        ss << inpName << ".y=(sqrt(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(sqrt(" << operand << ".z));\n";
                            ss << inpName << ".w=(sqrt(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(sqrt(" << operand << "))";
                    }
                    break;
                case UnaryOpOperation_RSQRT:
                    if(mVectorize) {
                        ss << inpName << ".x=(rsqrt(" << operand << ".x));\n";
                        ss << inpName << ".y=(rsqrt(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(rsqrt(" << operand << ".z));\n";
                            ss << inpName << ".w=(rsqrt(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(rsqrt(" << operand << "))";
                    }
                    break;
                case UnaryOpOperation_ABS:
                    if(mVectorize) {
                        ss << inpName << ".x=(fabs(" << operand << ".x));\n";
                        ss << inpName << ".y=(fabs(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(fabs(" << operand << ".z));\n";
                            ss << inpName << ".w=(fabs(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(fabs(" << operand << "))";
                    }                
                    break;
                case UnaryOpOperation_SIN:
                    if(mVectorize) {
                        ss << inpName << ".x=(sin(" << operand << ".x));\n";
                        ss << inpName << ".y=(sin(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(sin(" << operand << ".z));\n";
                            ss << inpName << ".w=(sin(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(sin(" << operand << "))";
                    }                
                    break;
                case UnaryOpOperation_COS:
                    if(mVectorize) {
                        ss << inpName << ".x=(cos(" << operand << ".x));\n";
                        ss << inpName << ".y=(cos(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(cos(" << operand << ".z));\n";
                            ss << inpName << ".w=(cos(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(cos(" << operand << "))";
                    }                
                    break;
                case UnaryOpOperation_ASIN:
                    if(mVectorize) {
                        ss << inpName << ".x=(asin(" << operand << ".x));\n";
                        ss << inpName << ".y=(asin(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(asin(" << operand << ".z));\n";
                            ss << inpName << ".w=(asin(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(asin(" << operand << "))";
                    }                      
                    break;
                case UnaryOpOperation_ACOS:
                    if(mVectorize) {
                        ss << inpName << ".x=(acos(" << operand << ".x));\n";
                        ss << inpName << ".y=(acos(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(acos(" << operand << ".z));\n";
                            ss << inpName << ".w=(acos(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(acos(" << operand << "))";
                    }                
                    break;
                case UnaryOpOperation_SIGN:
                    if(mVectorize) {
                        if(mPrecision == BackendConfig::Precision_Low) {
                            ss << inpName << ".x=((" << operand << ".x > (half)0.0) ? (half)1.0 : ((" << operand << ".x < (half)0.0) ? (half)(-1.0) : (half)0.0));\n";
                            ss << inpName << ".y=((" << operand << ".y > (half)0.0) ? (half)1.0 : ((" << operand << ".y < (half)0.0) ? (half)(-1.0) : (half)0.0))";
                        } else {
                            ss << inpName << ".x=((" << operand << ".x > 0.0) ? 1.0 : ((" << operand << ".x < 0.0) ? (-1.0) : 0.0));\n";
                            ss << inpName << ".y=((" << operand << ".y > 0.0) ? 1.0 : ((" << operand << ".y < 0.0) ? (-1.0) : 0.0))";
                            ss << ";\n";                           
                            ss << inpName << ".z=((" << operand << ".z > 0.0) ? 1.0 : ((" << operand << ".z < 0.0) ? (-1.0) : 0.0));\n";
                            ss << inpName << ".w=((" << operand << ".w > 0.0) ? 1.0 : ((" << operand << ".w < 0.0) ? (-1.0) : 0.0))";
                        }
                    } else {
                        if(mPrecision == BackendConfig::Precision_Low) {
                            ss << inpName << "=(" << operand << "> (half)0.0 ? (half)1.0 : (" << operand << "<(half)0.0 ? (half)(-1.0) : (half)0.0))";
                        } else {
                            ss << inpName << "=(" << operand << "> 0.0 ? 1.0 : (" << operand << "<0.0 ? (-1.0) : 0.0))";
                        }
                    }                    
                    break;
                case UnaryOpOperation_EXP:
                    if(mVectorize) {
                        ss << inpName << ".x=(exp(" << operand << ".x));\n";
                        ss << inpName << ".y=(exp(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(exp(" << operand << ".z));\n";
                            ss << inpName << ".w=(exp(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(exp(" << operand << "))";
                    }                
                    break;
                case UnaryOpOperation_NEG:
                    if(mVectorize) {
                        ss << inpName << ".x=(-(" << operand << ".x));\n";
                        ss << inpName << ".y=(-(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(-(" << operand << ".z));\n";
                            ss << inpName << ".w=(-(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(-(" << operand << "))";
                    }
                    break;
                case UnaryOpOperation_TAN:
                    if(mVectorize) {
                        ss << inpName << ".x=(tan(" << operand << ".x));\n";
                        ss << inpName << ".y=(tan(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(tan(" << operand << ".z));\n";
                            ss << inpName << ".w=(tan(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(tan(" << operand << "))";
                    }                       
                    break;
                case UnaryOpOperation_ATAN:
                    if(mVectorize) {
                        ss << inpName << ".x=(atan(" << operand << ".x));\n";
                        ss << inpName << ".y=(atan(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                          
                            ss << inpName << ".z=(atan(" << operand << ".z));\n";
                            ss << inpName << ".w=(atan(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(atan(" << operand << "))";
                    }                       
                    break;
                case UnaryOpOperation_CEIL:
                    if(mVectorize) {
                        ss << inpName << ".x=(ceil(" << operand << ".x));\n";
                        ss << inpName << ".y=(ceil(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(ceil(" << operand << ".z));\n";
                            ss << inpName << ".w=(ceil(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(ceil(" << operand << "))";
                    }                       
                    break;
                case UnaryOpOperation_LOG1P:
                    if(mVectorize) {
                        if(mPrecision == BackendConfig::Precision_Low) {
                            ss << inpName << ".x=(half)(log(1.0+(float)" << operand << ".x));\n";
                            ss << inpName << ".y=(half)(log(1.0+(float)" << operand << ".y))";
                        } else {
                            ss << inpName << ".x=(log(1.0+" << operand << ".x));\n";
                            ss << inpName << ".y=(log(1.0+" << operand << ".y))";
                            ss << ";\n";   
                            ss << inpName << ".z=(log(1.0+" << operand << ".z));\n";
                            ss << inpName << ".w=(log(1.0+" << operand << ".w))";
                        }
                    } else {
                        if(mPrecision == BackendConfig::Precision_Low) {
                            ss << inpName << "=(log((half)1.0+" << operand << "))";
                        } else {
                            ss << inpName << "=(log(1.0+" << operand << "))";
                        }
                    }
                    break;
                case UnaryOpOperation_FLOOR:
                    if(mVectorize) {
                        ss << inpName << ".x=(floor(" << operand << ".x));\n";
                        ss << inpName << ".y=(floor(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(floor(" << operand << ".z));\n";
                            ss << inpName << ".w=(floor(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(floor(" << operand << "))";
                    }                       
                    break;
                case UnaryOpOperation_ROUND:
                    if(mVectorize) {
                        ss << inpName << ".x=(round(" << operand << ".x));\n";
                        ss << inpName << ".y=(round(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(round(" << operand << ".z));\n";
                            ss << inpName << ".w=(round(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(round(" << operand << "))";
                    }                       
                    break;
                case UnaryOpOperation_SIGMOID:
                    if(mVectorize) {
                        if(mPrecision == BackendConfig::Precision_Low) {
                            ss << inpName << ".x=(half)(1.0/(1.0+(float)exp(-" << operand << ".x)));\n";
                            ss << inpName << ".y=(half)(1.0/(1.0+(float)exp(-" << operand << ".y)))";
                        } else {
                            ss << inpName << ".x=(1.0/(1.0+exp(-" << operand << ".x)));\n";
                            ss << inpName << ".y=(1.0/(1.0+exp(-" << operand << ".y)))";
                            ss << ";\n";                           
                            ss << inpName << ".z=(1.0/(1.0+exp(-" << operand << ".z)));\n";
                            ss << inpName << ".w=(1.0/(1.0+exp(-" << operand << ".w)))";
                        }
                    } else {
                        if(mPrecision == BackendConfig::Precision_Low) {
                            ss << inpName << "=(half)(1.0/(1.0+(float)exp(-" << operand << ")))";
                        } else {
                            ss << inpName << "=(1.0/(1.0+exp(-" << operand << ")))";
                        }
                    }
                    break;
                case UnaryOpOperation_TANH:
                    if(mVectorize) {
                        ss << inpName << ".x=(tanh(" << operand << ".x));\n";
                        ss << inpName << ".y=(tanh(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(tanh(" << operand << ".z));\n";
                            ss << inpName << ".w=(tanh(" << operand << ".w));";
                        }
                    } else {
                        ss << inpName << "=(tanh(" << operand << "))";
                    }
                    break;
                case UnaryOpOperation_RECIPROCAL:
                    if(mVectorize) {
                        if(mPrecision == BackendConfig::Precision_Low) {
                            ss << inpName << ".x=(half)(1.0/(float)" << operand << ".x);\n";
                            ss << inpName << ".y=(half)(1.0/(float)" << operand << ".y)";
                        } else {
                            ss << inpName << ".x=(1.0/" << operand << ".x);\n";
                            ss << inpName << ".y=(1.0/" << operand << ".y)";
                            ss << ";\n";                           
                            ss << inpName << ".z=(1.0/" << operand << ".z);\n";
                            ss << inpName << ".w=(1.0/" << operand << ".w)";
                        }
                    } else {
                        if(mPrecision == BackendConfig::Precision_Low) {
                            ss << inpName << "=(half)(1.0/(float)" << operand << ")";
                        } else {
                            ss << inpName << "=(1.0/" << operand << ")";
                        }
                    }                
                    break;
                case UnaryOpOperation_LOG:
                    if(mVectorize) {
                        ss << inpName << ".x=(log(" << operand << ".x));\n";
                        ss << inpName << ".y=(log(" << operand << ".y))";
                        if(mPrecision != BackendConfig::Precision_Low) {
                            ss << ";\n";                           
                            ss << inpName << ".z=(log(" << operand << ".z));\n";
                            ss << inpName << ".w=(log(" << operand << ".w))";
                        }
                    } else {
                        ss << inpName << "=(log(" << operand << "))";
                    }                       
                    break;
                case UnaryOpOperation_GELU:
                    if(mVectorize) {
                        if(mPrecision == BackendConfig::Precision_Low) {
                            ss << inpName << ".x=(half)((1.0f + tanh(0.79788458f * (0.044715f * (float)" << operand <<  ".x*(float)" << operand <<  ".x*(float)" <<  operand <<  ".x+(float)" <<  operand + ".x))) * (float)"  << operand << ".x* 0.5f);\n";
                            ss << inpName << ".y=(half)((1.0f + tanh(0.79788458f * (0.044715f * (float)" << operand <<  ".y*(float)" << operand <<  ".y*(float)" <<  operand <<  ".y+(float)" <<  operand + ".y))) * (float)"  << operand << ".y* 0.5f)";
                        } else {
                            ss << inpName << ".x=((1.0f + tanh(0.79788458f * (0.044715f * " << operand <<  ".x*" << operand <<  ".x*" <<  operand <<  ".x+" <<  operand + ".x))) * "  << operand << ".x* 0.5f);\n";
                            ss << inpName << ".y=((1.0f + tanh(0.79788458f * (0.044715f * " << operand <<  ".y*" << operand <<  ".y*" <<  operand <<  ".y+" <<  operand + ".y))) * "  << operand << ".y* 0.5f)";
                            ss << ";\n";   
                            ss << inpName << ".z=((1.0f + tanh(0.79788458f * (0.044715f * " << operand <<  ".z*" << operand <<  ".z*" <<  operand <<  ".z+" <<  operand + ".z))) * "  << operand << ".z* 0.5f);\n";
                            ss << inpName << ".w=((1.0f + tanh(0.79788458f * (0.044715f * " << operand <<  ".w*" << operand <<  ".w*" <<  operand <<  ".w+" <<  operand + ".w))) * "  << operand << ".w* 0.5f)";
                        }
                    } else {
                        if(mPrecision == BackendConfig::Precision_Low) {
                            ss << inpName << "=(half)((1.0f + tanh(0.79788458f * (0.044715f * (float)" << operand <<  "*(float)" << operand <<  "*(float)" <<  operand <<  "+(float)" <<  operand + "))) * (float)"  << operand << "* 0.5f)";
                        } else {
                            ss << inpName << "=((1.0f + tanh(0.79788458f * (0.044715f * " << operand <<  "*" << operand <<  "*" <<  operand <<  "+" <<  operand + "))) * "  << operand << "* 0.5f)";
                        }
                    }                
                    break;
                case UnaryOpOperation_GELU_STANDARD:
                    if(mVectorize) {
                        if(mPrecision == BackendConfig::Precision_Low) {
                            ss << inpName << ".x=(half)((erf((float)" << operand << ".x*0.7071067932881648f)+1.f)*(float)" << operand << ".x*0.5f);\n";
                            ss << inpName << ".y=(half)((erf((float)" << operand << ".y*0.7071067932881648f)+1.f)*(float)" << operand << ".y*0.5f)";
                        } else {
                            ss << inpName << ".x=((erf(" << operand << ".x*0.7071067932881648f)+1.f)*" << operand << ".x*0.5f);\n";
                            ss << inpName << ".y=((erf(" << operand << ".y*0.7071067932881648f)+1.f)*" << operand << ".y*0.5f)";
                            ss << ";\n";   
                            ss << inpName << ".z=((erf(" << operand << ".z*0.7071067932881648f)+1.f)*" << operand << ".z*0.5f);\n";
                            ss << inpName << ".w=((erf(" << operand << ".w*0.7071067932881648f)+1.f)*" << operand << ".w*0.5f)";
                        }
                    } else {
                        if(mPrecision == BackendConfig::Precision_Low) {
                            ss << inpName << "=(half)((erf((float)" << operand << "*0.7071067932881648f)+1.f)*(float)" << operand << "*0.5f)";
                        } else {
                            ss << inpName << "=((erf(" << operand << "*0.7071067932881648f)+1.f)*" << operand << "*0.5f)";
                        }
                    }
                    break;
                default:
                    MNN_PRINT("Error: CUDA CodeGen not support Unary type:%d\n", type);
                    break;
            }
            break;
        }
        case MNN::OpType_ReLU6:
        {
            auto operand = inputs[0];
            auto relu6 = op->main_as_Relu6();
            float minv = relu6->minValue();
            float maxv = relu6->maxValue();
            if(mVectorize) {
                ss << inpName << ".x=fmin(fmax(" << operand << ".x," << numval(minv) << "), " << numval(maxv) << ");\n";
                ss << inpName << ".y=fmin(fmax(" << operand << ".y," << numval(minv) << "), " << numval(maxv) << ")";
                if(mPrecision != BackendConfig::Precision_Low) {
                    ss << ";\n";                   
                    ss << inpName << ".z=fmin(fmax(" << operand << ".z," << numval(minv) << "), " << numval(maxv) << ");\n";
                    ss << inpName << ".w=fmin(fmax(" << operand << ".w," << numval(minv) << "), " << numval(maxv) << ")";
                }
            } else {
                ss << inpName << "=fmin(fmax(" << operand << "," << numval(minv) << "), " << numval(maxv) << ")";
            }
            break;
        }
        case MNN::OpType_ReLU:
        {
            auto operand = inputs[0];
            auto relu = op->main_as_Relu();
            float slope = relu->slope();
            if(mVectorize) {
                ss << inpName << ".x=fmax(" << operand << ".x," << numval(0) << ");\n";
                ss << inpName << ".y=fmax(" << operand << ".y," << numval(0) << ")";
                if(mPrecision != BackendConfig::Precision_Low) {
                    ss << ";\n";                 
                    ss << inpName << ".z=fmax(" << operand << ".z," << numval(0) << ");\n";
                    ss << inpName << ".w=fmax(" << operand << ".w," << numval(0) << ")";
                }
            } else {
                ss << inpName << "=fmax(" << operand << "," << numval(0) << ")";
            }
            break;
        }
        case MNN::OpType_Raster:
        {   
            auto operand = inputs[0];
            ss << inpName << "=(" << operand << ")";
            break;
        }
        default:
            break;
    }
    return ss.str();
}
std::string CUDATarget::load(const std::string& base, const std::string& offset, const Command* cmd, std::string& inpName) {
    if(cmd->op->type() == MNN::OpType_Raster) {
        OpCommonUtils::TensorConvertParameter singleConvert;
        auto input = cmd->outputs[0];
        OpCommonUtils::rasterInputReset(cmd->inputs, cmd->outputs[0]);
        singleConvert.type = 0;
        auto des = TensorUtils::getDescribe(input);
        if(des->regions.size() == 1) {
            OpCommonUtils::turnRegion2Convert(des->regions[0], cmd->outputs[0], singleConvert);
            if (singleConvert.type > 0) {
                auto realInput = TensorUtils::getDescribe(input)->regions[0].origin;
                auto sourceFormat = TensorUtils::getDescribe(realInput)->dimensionFormat;
                if (MNN_DATA_FORMAT_NC4HW4 == sourceFormat) { // NC4HW4 -> NCHW/NHWC is Supported!
                    std::string res;
                    int srcBatch = singleConvert.batch;
                    int srcChannel = singleConvert.channel;
                    int srcArea = singleConvert.area;
                    res += type() + inpName + ";\n";
                    if (singleConvert.type == 1) {// NCHW
                        res += ("    {\n\tint idx_area = " + offset + "%" + "area;\n");
                        res += ("\tint idx_bc = " + offset + "/" + "area;\n");
                        res += ("\tint idx_channel = idx_bc \% channel;\n");
                        res += ("\tint idx_batch = idx_bc / channel;\n");
                        res += ("\tint inp_index = (idx_batch * area + idx_area) * channel_pack + idx_channel;\n\t");
                        res += inpName + "=(" + base + "[inp_index]);\n    }";
                        return res;
                    } else if (singleConvert.type == 2) {// NHWC
                        if(srcChannel % 8 == 0) {
                            res += ("    {\n\tint inp_index = " + offset + ";\n\t");
                        } else {
                            // res += ("    {\n\tint idx_channel, idx_area, idx_batch, tmp;\n");
                            // res += ("\tchannel_d->divmod(" + offset + ", tmp, idx_channel);\n");
                            // res += ("\tarea_d->divmod(tmp, idx_batch, idx_area);\n");
                            // res += ("\tint inp_index = (idx_batch * area + idx_area) * channel_pack + idx_channel;\n\t");
                            res += ("    {\n\tint idx_channel = " + offset + "%" + "channel;\n");
                            res += ("\tint idx_ba = " + offset + "/" + "channel;\n");
                            res += ("\tint idx_area = idx_ba \% area;\n");
                            res += ("\tint idx_batch = idx_ba / area;\n");
                            res += ("\tint inp_index = (idx_batch * area + idx_area) * channel_pack + idx_channel;\n\t");
                        }                       
                        if(mVectorize) {
                            if(mPrecision == BackendConfig::Precision_Low) {
                                res += inpName + "=(((half2 *)" + base + ")[inp_index]);\n    }";
                            } else {
                                res += inpName + "=(((float4 *)" + base + ")[inp_index]);\n    }";
                            }
                        } else {
                            res += inpName + "=(" + base + "[inp_index]);\n    }";
                        }
                        return res;
                    }
                }
            }
        }
    }
    if(mVectorize) {
        if(mPrecision == BackendConfig::Precision_Low) {
            return  type() + inpName + "=(((half2 *)" + base + ")[" + offset + "])";
        }
        return  type() + inpName + "=(((float4 *)" + base + ")[" + offset + "])";
    }
    return  type() + inpName + "=(" + base + "[" + offset + "])";
}
std::string CUDATarget::loadscalar(const std::string& base, std::string& inpName) {
    if(mVectorize) {
        if(mPrecision == BackendConfig::Precision_Low) {
            return  type() + inpName + "=(((half2 *)" + base + ")[0])";
        }
        return  type() + inpName + "=(((float4 *)" + base + ")[0])";
    }
    return  type() + inpName + "=(" + base + "[0])";
}
std::string CUDATarget::store(const std::string base, const std::string& offset, const std::string& data) {
    if(mVectorize) {
        if(mPrecision == BackendConfig::Precision_Low) {
            return "((half2 *)" + base + ")[" + offset + "] = " + data + ";\n";
        }
        return "((float4 *)" + base + ")[" + offset + "] = " + data + ";\n";
    }
    return base + "[" + offset + "] = " + data + ";\n";
}

std::string CUDATarget::proto(const std::string& name, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, bool hasSingleConvertRaster) {
    std::stringstream proto;
    std::string begin = "extern \"C\" __global__ void ";
    mKernelBeginSize = begin.size();
    proto << begin << "(";//extern \"C\" 
    if(mPrecision == BackendConfig::Precision_Low) {
        for (auto& input : inputs) {
            proto << "const half* " << input << ", ";
        }
        for (auto& output : outputs) {
            proto << "half* " << output << ", ";
        }
    } else {
        for (auto& input : inputs) {
            proto << "const float* " << input << ", ";
        }
        for (auto& output : outputs) {
            proto << "float* " << output << ", ";
        }
    }

    proto << "const size_t count";
    if (hasSingleConvertRaster) {
        proto << ", const size_t batch, ";
        proto << "const size_t area, ";
        proto << "const size_t channel, ";
        proto << "const size_t channel_pack";
        // proto << "DivModFast* area_d,";
        // proto << "DivModFast* channel_d";
    }
    proto << ")";


    return proto.str();
}
} // MNN

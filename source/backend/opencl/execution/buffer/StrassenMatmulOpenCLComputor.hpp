//
//  StrassenMatmulComputor.hpp
//  MNN
//
//  Created by MNN on 2024/08/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef StrassenMatmulOpenCLComputor_hpp
#define StrassenMatmulOpenCLComputor_hpp

#include "core/BufferAllocator.hpp"
#include "core/Backend.hpp"
#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {
/**
 Based on
 Boyer, B., Dumas, J.-G., Pernet, C., & Zhou, W. (2007). Memory efficient scheduling of Strassen-Winogradʼs matrix multiplication algorithm. Proceedings of the 2009 international symposium on Symbolic and algebraic computation ISSAC 09, 55. ACM Press. Retrieved from http://arxiv.org/abs/0707.2347
 
 Use Table 2
 */
class StrassenMatrixComputor {
public:
    StrassenMatrixComputor(Backend* bn, int maxDepth);
    virtual ~StrassenMatrixComputor();
    
    ErrorCode onEncode(int e, int l, int h, int as, int bs, int cs, const cl::Buffer AT, const cl::Buffer BT, cl::Buffer CT, bool useBias, const cl::Buffer Bias);
    
    void onExecute();
    
    void onReset();
private:
    struct MatrixInfo {
        int stackIndex;
        int offsetBytes;
        int lineStrideBytes;
    };
    
    /* postType:
     0 --> without post process
     1 --> with bias (one dimension)
     2 --> with feature map D to eltwise add ( Y = X + D)
     3 --> with feature map D to eltwise sub ( Y = X - D)
     4 --> with feature map D to eltwise sub and get negative( Y = D - X)
     */
    ErrorCode _generateMatMul(int e, int l, int h, const MatrixInfo& AT, const MatrixInfo& BT, const MatrixInfo& CT, const MatrixInfo& COT, int currentDepth, int postType = 0);
    ErrorCode _generateBasicMatMul(int e, int l, int h, const MatrixInfo& AT, const MatrixInfo& BT, const MatrixInfo& CT, const MatrixInfo& COT, int postType, Unit& unit);
    
    ErrorCode _generateBinary(cl::Buffer ptrC, cl::Buffer ptrA, cl::Buffer ptrB, int offsetC, int offsetA, int offsetB, int elementStrideC, int elementStrideA, int elementStrideB, int width, int height, bool isAdd, Unit& unit);

    ErrorCode _generateCFunction(cl::Buffer ptrC, int offsetC, int elementStrideC, cl::Buffer ptrA, int width, int height, Unit& unit);
    
private:
    std::vector<Unit> mUnits;
    int mMaxDepth;
    OpenCLBackend* mOpenCLBackend;
    int mM, mN, mK;
    std::vector<cl::Buffer> mStack;
    int mBytes = 4;
};
} // namespace MNN
}
#endif /* StrassenMatmulOpenCLComputor_hpp */
#endif

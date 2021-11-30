//
//  WinogradOptFunctionInt8.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef WinogradOptFunctionInt8_hpp
#define WinogradOptFunctionInt8_hpp

namespace MNN {
struct CoreInt8Functions;
class WinogradFunctionInt8 {
public:
    typedef void(*SrcTransXFunc)(const int8_t* srcStart, int8_t* dstStart, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countC4, size_t xC, size_t unit);
    typedef void(*SrcTransYFunc)(const int8_t* srcStart, int8_t* dstStart, size_t srcYStep, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countC4, size_t xC);
    typedef void(*SrcTrans2Func)(const int8_t* srcStart, int8_t* dstStart, size_t srcYStep, size_t srcZStep, size_t dstXStep, size_t dstZStep, size_t countC4, size_t xC, size_t unit);
    static SrcTransXFunc chooseSourceTransformX(int alpha, int inPack, int outPack);
    static SrcTransYFunc chooseSourceTransformY(int alpha, int inPack, int outPack);
    static SrcTrans2Func chooseSourceTransform2(int alpha, int inPack, int outPack);
};
}

#endif // WinogradOptFunctionInt8_hpp

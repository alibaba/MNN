//
//  CPUBinary.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUBinary.hpp"
#include <math.h>
#include <algorithm>
#include "CPUBackend.hpp"
#include "compute/CommonOptFunction.h"
#include "compute/ConvOpt.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/OpCommonUtils.hpp"
namespace MNN {
#define MAX_DIM 6
CPUBinaryInt::CPUBinaryInt(Backend* b, int32_t type) : MNN::Execution(b), mType(type) {
    // nothing to do
}
CPUBinaryFloat::CPUBinaryFloat(Backend* b, int32_t type) : MNN::Execution(b), mType(type) {
    // nothing to do
}

ErrorCode CPUBinaryFloat::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(1 == outputs.size());
    const int input0DataCount = inputs[0]->elementSize();
    const int input1DataCount = inputs[1]->elementSize();
    const int outputDataCount = outputs[0]->elementSize();
    int maxCount = input0DataCount > input1DataCount ?  input0DataCount : input1DataCount;
    mElementProc = nullptr;
    mSupportScale = false;
    if (outputs[0]->getType().code != halide_type_float || maxCount < 4 || (outputDataCount > input0DataCount && outputDataCount > input1DataCount)) {
        // Can't optimize
        return NO_ERROR;
    }
    auto eleProc = mElementProc;// Set nullptr for begin
    switch (mType) {
        case BinaryOpOperation_MUL:
            eleProc = MNNMatrixProdCommon;
            break;
        case BinaryOpOperation_ADD:
            eleProc = MNNMatrixAddCommon;
            break;
        case BinaryOpOperation_MAXIMUM:
            eleProc = MNNMatrixMaxCommon;
            break;
        case BinaryOpOperation_SUB:
            eleProc = MNNMatrixSubCommon;
            break;
        default:
            break;
    }
    if (input1DataCount == input0DataCount) {
        mOutside = 1;
        mInside = input0DataCount;
        mElementProc = eleProc;
        return NO_ERROR;
    }
    if (input1DataCount == 1 || input0DataCount == 1) {
        mAxis = 1;
        mOutside = 1;
        switch (mType) {
            case BinaryOpOperation_MUL:
            case BinaryOpOperation_ADD:
            case BinaryOpOperation_SUB:
                mSupportScale = true;
                break;
            default:
                break;
        }
        return NO_ERROR;
    }
    if (nullptr == eleProc) {
        return NO_ERROR;
    }
    // For AddBias / Mul Sqrt
    int dims[MAX_DIM];
    int stride[MAX_DIM];
    int iStride0[MAX_DIM];
    int iStride1[MAX_DIM];
    const Tensor* input0 = inputs[0];
    const Tensor* input1 = inputs[1];
    const Tensor* output = outputs[0];
    if (input0DataCount < input1DataCount) {
        input0 = inputs[1];
        input1 = inputs[0];
    }
    OpCommonUtils::broastCastComputeDim(dims, stride, iStride0, iStride1, input0, input1, output);
    int breakPos = -1;
    for (int i=0; i<MAX_DIM; ++i) {
        if (iStride1[i] > 0) {
            if (breakPos >= 0) {
                // Failed to optmize
                return NO_ERROR;
            }
            breakPos = i;
        }
    }
    MNN_ASSERT(breakPos >= 0);
    //FUNC_PRINT(breakPos);
    mOutside = 1;
    mInside = 1;
    for (int i=0; i<breakPos; ++i) {
        mOutside *= dims[i];
    }
    mAxis = dims[breakPos];
    for (int i=breakPos+1; i<MAX_DIM; ++i) {
        mInside *= dims[i];
    }
    // Serveral Machine need memory 4 * sizeof(float) align
    if (1 == mInside && mAxis >= 4) {
        mElementProc = eleProc;
        //MNN_PRINT("Open Optimize\n");
    } else if (BinaryOpOperation_MAXIMUM != mType && mInside >= 4) {
        mSupportScale = true;
    }
    //MNN_PRINT("%d, %d, %d\n", mInside, mAxis, mOutside);
    return NO_ERROR;
}

template <typename Tin, typename Tout, typename Func>
static ErrorCode _binaryOp(Tensor* input0, Tensor* input1, Tensor* output) {
    Func f;
    const int input0DataCount = input0->elementSize();
    const int input1DataCount = input1->elementSize();
    const Tin* input0Data = input0->host<Tin>();
    const Tin* input1Data = input1->host<Tin>();
    Tout* outputData      = output->host<Tout>();

    if (input0DataCount == 1) { // data count == 1, not only mean scalar input, maybe of shape (1, 1, 1, ...,1)
        for (int i = 0; i < input1DataCount; i++) {
            outputData[i] = static_cast<Tout>(f(input0Data[0], input1Data[i]));
        }
    } else if (input1DataCount == 1) {
        for (int i = 0; i < input0DataCount; i++) {
            outputData[i] = static_cast<Tout>(f(input0Data[i], input1Data[0]));
        }
    } else { // both input contains more than one element，which means no scalar input
        bool sameShape = true;
        {
            if (input0->dimensions() == input1->dimensions()) {
                for (int i = 0; i < input0->buffer().dimensions; i++) {
                    if (input0->buffer().dim[i].extent != input1->buffer().dim[i].extent) {
                        sameShape = false;
                        break;
                    }
                }
            }
            else {
                sameShape = false;
            }
        }
        if (sameShape) { // two inputs have the same shape, apply element-wise operation
            for (int i = 0; i < input0DataCount; i++) {
                outputData[i] = static_cast<Tout>(f(input0Data[i], input1Data[i]));
            }
        } else { // not the same shape, use broadcast
            MNN_ASSERT(output->dimensions() <= MAX_DIM);
            int dims[MAX_DIM];
            int stride[MAX_DIM];
            int iStride0[MAX_DIM];
            int iStride1[MAX_DIM];
            OpCommonUtils::broastCastComputeDim(dims, stride, iStride0, iStride1, input0, input1, output);
            for (int w = 0; w < dims[5]; ++w) {
                auto ow  = outputData + w * stride[5];
                auto i0w = input0Data + w * iStride0[5];
                auto i1w = input1Data + w * iStride1[5];
#define PTR(x, y, i)                      \
    auto o##x  = o##y + x * stride[i];    \
    auto i0##x = i0##y + x * iStride0[i]; \
    auto i1##x = i1##y + x * iStride1[i]

                for (int v = 0; v < dims[4]; ++v) {
                    PTR(v, w, 4);
                    for (int u = 0; u < dims[3]; ++u) {
                        PTR(u, v, 3);
                        for (int z = 0; z < dims[2]; ++z) {
                            PTR(z, u, 2);
                            for (int y = 0; y < dims[1]; ++y) {
                                PTR(y, z, 1);
                                for (int x = 0; x < dims[0]; ++x) {
                                    PTR(x, y, 0);
                                    *ox = static_cast<Tout>(f(*i0x, *i1x));
                                }
                            }
                        }
                    }
                }
            }
#undef MAX_DIM
#undef PTR
        }
        // broadcast-capable check is done in compute size
    }

    return NO_ERROR;
}

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMax : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return std::max(x, y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMin : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return std::min(x, y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMul : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x * y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryAdd : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x + y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinarySub : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x - y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryRealDiv : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x / y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryMod : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x - x / y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryGreater : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x > y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryLess : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x < y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryGreaterEqual : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x >= y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryLessEqual : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x <= y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryEqual : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x == y) ? 1 : 0);
    }
};
template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryFloorDiv : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return floor(x / y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryFloorMod : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return x - floor(x / y) * y;
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinarySquaredDifference : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (x - y) * (x - y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryPow : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return pow(x, y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryAtan2 : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return atan(x / y);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryLogicalOr : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x || y) ? 1 : 0);
    }
};

template <typename _Arg1, typename _Arg2, typename _ErrorCode>
struct BinaryNotEqual : std::binary_function<_Arg1, _Arg2, _ErrorCode> {
    _ErrorCode operator()(const _Arg1& x, const _Arg2& y) const {
        return (_ErrorCode)((x != y) ? 1 : 0);
    }
};

ErrorCode CPUBinaryFloat::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto input1 = inputs[1];
    auto output = outputs[0];
    
    if (nullptr != mElementProc || mSupportScale) {
        auto numberThread = ((CPUBackend*)backend())->threadNumber();
        auto i1Size = input->elementSize();
        auto i2Size = input1->elementSize();
        bool swap = false;
        if (i1Size < i2Size) {
            auto temp = i2Size;
            i2Size = i1Size;
            i1Size = temp;
            input = inputs[1];
            input1 = inputs[0];
            swap = true;
        }
        auto size = i1Size;
        auto schedule = ((CPUBackend*)backend())->multiThreadDivide(size);
        int sizeDivide = schedule.first;
        int scheduleNumber = schedule.second;
        if (nullptr != mElementProc) {
            if (mOutside == 1) {
                MNN_CONCURRENCY_BEGIN(tId, scheduleNumber) {
                    int start = sizeDivide * (int)tId;
                    int realSize = sizeDivide;
                    if (tId == scheduleNumber -1 ) {
                        realSize = size - start;
                    }
                    if (realSize > 0) {
                        mElementProc(output->host<float>() + start, input->host<float>() + start, input1->host<float>() + start, realSize, 0, 0, 0, 1);
                    }
                }
                MNN_CONCURRENCY_END();
            } else {
                MNN_CONCURRENCY_BEGIN(tId, numberThread) {
                    for (int y = tId; y < mOutside; y+=numberThread) {
                        mElementProc(output->host<float>() + y * mAxis, input->host<float>() + y * mAxis, input1->host<float>(), mAxis, 0, 0, 0, 1);
                    }
                }
                MNN_CONCURRENCY_END();
            }
        } else {
            if (mOutside == 1 && mAxis == 1) {
                float* inputPtr = input->host<float>();
                float scalar = input1->host<float>()[0];
                float scale = scalar;
                float bias = 0.0f;
                switch (mType) {
                    case BinaryOpOperation_ADD:
                        scale = 1.0f;
                        bias = scalar;
                        break;
                    case BinaryOpOperation_SUB:
                        if (!swap) {
                            scale = 1.0f;
                            bias = -scalar;
                        } else {
                            scale = -1.0f;
                            bias = scalar;
                        }
                        break;
                    default:
                        break;
                }

                MNN_CONCURRENCY_BEGIN(tId, scheduleNumber) {
                    int start = sizeDivide * (int)tId;
                    int realSize = sizeDivide;
                    if (tId == scheduleNumber -1 ) {
                        realSize = size - start;
                    }
                    if (realSize > 0) {
                        MNNScaleAndAddBiasScalar(output->host<float>() + start, inputPtr + start, bias, scale, realSize);
                    }
                }
                MNN_CONCURRENCY_END();
            } else {
                float* inputPtr = input->host<float>();
                float* input1Ptr = input1->host<float>();
                auto total = mOutside * mAxis;
                MNN_CONCURRENCY_BEGIN(tId, numberThread) {
                    for (int index = tId; index < total; index += numberThread) {
                        auto axis = index % mAxis;
                        float scalar = input1Ptr[axis];
                        float scale = scalar;
                        float bias = 0.0f;
                        switch (mType) {
                            case BinaryOpOperation_ADD:
                                scale = 1.0f;
                                bias = scalar;
                                break;
                            case BinaryOpOperation_SUB:
                                if (!swap) {
                                    scale = 1.0f;
                                    bias = -scalar;
                                } else {
                                    scale = -1.0f;
                                    bias = scalar;
                                }
                                break;
                            default:
                                break;
                        }
                        MNNScaleAndAddBiasScalar(output->host<float>() + mInside * index, inputPtr +  mInside * index, bias, scale, mInside);
                    }
                }
                MNN_CONCURRENCY_END();
            }

        }
        return NO_ERROR;
    }

    switch (mType) {
        case BinaryOpOperation_MUL:
            _binaryOp<float, float, BinaryMul<float, float, float>>(input, input1, output);
            break;
        case BinaryOpOperation_ADD:
            _binaryOp<float, float, BinaryAdd<float, float, float>>(input, input1, output);
            break;
        case BinaryOpOperation_SUB:
            _binaryOp<float, float, BinarySub<float, float, float>>(input, input1, output);
            break;

        case BinaryOpOperation_REALDIV:
            _binaryOp<float, float, BinaryRealDiv<float, float, float>>(input, input1, output);
            break;
        case BinaryOpOperation_MINIMUM:
            _binaryOp<float, float, BinaryMin<float, float, float>>(input, input1, output);
            break;
        case BinaryOpOperation_MAXIMUM:
            _binaryOp<float, float, BinaryMax<float, float, float>>(input, input1, output);
            break;
        case BinaryOpOperation_GREATER:
            _binaryOp<float, int32_t, BinaryGreater<float, float, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_LESS:
            _binaryOp<float, float, BinaryLess<float, float, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_LESS_EQUAL:
            _binaryOp<float, float, BinaryLessEqual<float, float, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_GREATER_EQUAL:
            _binaryOp<float, float, BinaryGreaterEqual<float, float, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_EQUAL:
            _binaryOp<float, float, BinaryEqual<float, float, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_FLOORDIV:
            _binaryOp<float, float, BinaryFloorDiv<float, float, float>>(input, input1, output);
            break;
        case BinaryOpOperation_FLOORMOD:
            _binaryOp<float, float, BinaryFloorMod<float, float, float>>(input, input1, output);
            break;
        case BinaryOpOperation_POW:
            _binaryOp<float, float, BinaryPow<float, float, float>>(input, input1, output);
            break;
        case BinaryOpOperation_SquaredDifference:
            _binaryOp<float, float, BinarySquaredDifference<float, float, float>>(input, input1, output);
            break;
        case BinaryOpOperation_ATAN2:
            _binaryOp<float, float, BinaryAtan2<float, float, float>>(input, input1, output);
            break;
        case BinaryOpOperation_NOTEQUAL:
            _binaryOp<float, float, BinaryNotEqual<float, float, float>>(input, input1, output);
            break;
        case BinaryOpOperation_MOD:
            _binaryOp<float, float, BinaryMod<float, float, float>>(input, input1, output);
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    return NO_ERROR;
}

ErrorCode CPUBinaryInt::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto input1 = inputs[1];
    auto output = outputs[0];
    switch (mType) {
        case BinaryOpOperation_MUL:
            _binaryOp<int32_t, int32_t, BinaryMul<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_ADD:
            _binaryOp<int32_t, int32_t, BinaryAdd<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_SUB:
            _binaryOp<int32_t, int32_t, BinarySub<int32_t, int32_t, int32_t>>(input, input1, output);
            break;

        case BinaryOpOperation_REALDIV:
            _binaryOp<int32_t, int32_t, BinaryRealDiv<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_MINIMUM:
            _binaryOp<int32_t, int32_t, BinaryMin<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_MAXIMUM:
            _binaryOp<int32_t, int32_t, BinaryMax<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_GREATER:
            _binaryOp<int32_t, int32_t, BinaryGreater<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_LESS:
            _binaryOp<int32_t, int32_t, BinaryLess<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_LESS_EQUAL:
            _binaryOp<int32_t, int32_t, BinaryLessEqual<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_GREATER_EQUAL:
            _binaryOp<int32_t, int32_t, BinaryGreaterEqual<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_EQUAL:
            _binaryOp<int32_t, int32_t, BinaryEqual<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_FLOORDIV:
            _binaryOp<int32_t, int32_t, BinaryFloorDiv<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_FLOORMOD:
            _binaryOp<int32_t, int32_t, BinaryFloorMod<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_SquaredDifference:
            _binaryOp<int32_t, int32_t, BinarySquaredDifference<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_LOGICALOR:
            _binaryOp<int32_t, int32_t, BinaryLogicalOr<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_NOTEQUAL:
            _binaryOp<int32_t, int32_t, BinaryNotEqual<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        case BinaryOpOperation_MOD:
            _binaryOp<int32_t, int32_t, BinaryMod<int32_t, int32_t, int32_t>>(input, input1, output);
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    return NO_ERROR;
}

class CPUBinaryCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto dataType   = outputs[0]->getType();
        int32_t type = op->main_as_BinaryOp()->opType();
        if (dataType.bits == 32) {
            if (dataType.code == halide_type_int) {
                return new CPUBinaryInt(backend, type);
            }
            if (dataType.code == halide_type_float) {
                return new CPUBinaryFloat(backend, type);
            }
        }
        return nullptr;
    }
};

REGISTER_CPU_OP_CREATOR(CPUBinaryCreator, OpType_BinaryOp);

} // namespace MNN

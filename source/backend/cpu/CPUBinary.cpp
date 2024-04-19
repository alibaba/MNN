//
//  CPUBinary.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUBinary.hpp"
#include "CPUBinaryInt8.hpp"
#include "CPUBackend.hpp"
#include "compute/CommonOptFunction.h"
#include "compute/ConvOpt.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/OpCommonUtils.hpp"
#include "BinaryUtils.hpp"
#include "math/Vec.hpp"
using Vec4 = MNN::Math::Vec<float, 4>;
using Vec4Int = MNN::Math::Vec<int32_t, 4>;

namespace MNN {

ErrorCode CPUBinary::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input0DataCount = TensorUtils::getRawSize(inputs[0]);
    auto input1DataCount = TensorUtils::getRawSize(inputs[1]);
    if (input1DataCount == input0DataCount) {
        mNeedBroadcastIndex = -1;
    } else if (input0DataCount == 1) {
        mNeedBroadcastIndex = 0;
    } else {
        mNeedBroadcastIndex = 1;
    }
    mTotalSize = ((CPUBackend*)backend())->getTensorSize(outputs[0]);
    
    if(mActivationType == 1 && outputs[0]->getType().code == halide_type_float) {
        mActivationExe.reset(new CPURelu(backend(), 0.0));
        mActivationExe->onResize(outputs, outputs);
    }
    return NO_ERROR;
}

ErrorCode CPUBinary::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto input1 = inputs[1];
    auto output = outputs[0];
    
    auto schedule = ((CPUBackend*)backend())->multiThreadDivide(mTotalSize);
    auto input0Ptr = input->host<uint8_t>();
    auto input1Ptr = input1->host<uint8_t>();
    
    auto outputPtr = outputs[0]->host<uint8_t>();

    int inpBytes = input->getType().bytes();
    int outBytes = output->getType().bytes();
    if (halide_type_float == input->getType().code) {
        inpBytes = static_cast<CPUBackend*>(backend())->functions()->bytes;
    }
    if (halide_type_float == output->getType().code) {
        outBytes = static_cast<CPUBackend*>(backend())->functions()->bytes;
    }
    auto precision = static_cast<CPUBackend*>(backend())->precisionMode();
    MNN_CONCURRENCY_BEGIN(tId, schedule.second) {
        int start = schedule.first * (int)tId;
        int realSize = schedule.first;
        if (tId == schedule.second -1 ) {
            realSize = mTotalSize - start;
        }
        if (realSize > 0) {
            auto inp0 = input0Ptr + start * inpBytes;
            auto inp1 = input1Ptr + start * inpBytes;
            if (mNeedBroadcastIndex == 0) {
                inp0 = input0Ptr;
            } else if (mNeedBroadcastIndex == 1) {
                inp1 = input1Ptr;
            }
            auto out = outputPtr + start * outBytes;
            mProc(out, inp0, inp1, realSize, mNeedBroadcastIndex);
            if(mActivationType == 1 && output->getType().code == halide_type_int) {
                for(int i=0; i<realSize; i++) {
                    auto val = ((int32_t *)out)[i];
                    auto res = val > 0 ? val : 0;
                    ((int32_t *)out)[i] = res;
                }
            }
        }
    }
    MNN_CONCURRENCY_END();
    
    if(mActivationType == 1 && output->getType().code == halide_type_float) {
        mActivationExe->onExecute(outputs, outputs);
    }
    return NO_ERROR;
}

MNNBinaryExecute CPUBinary::selectForFloat(int type) {
    auto vecFunction = selectVector<Vec4, 4, float>(type);
    if (nullptr != vecFunction) {
        return vecFunction;
    }
    switch (type) {
        case BinaryOpOperation_REALDIV:
            return execute<float, float, BinaryRealDiv<float, float, float>>;
        case BinaryOpOperation_FLOORDIV:
            return execute<float, float, BinaryFloorDiv<float, float, float>>;
        case BinaryOpOperation_FLOORMOD:
            return execute<float, float, BinaryFloorMod<float, float, float>>;
        case BinaryOpOperation_NOTEQUAL:
            return execute<float, int32_t, BinaryNotEqual<float, float, int32_t>>;
        case BinaryOpOperation_POW:
            return execute<float, float, BinaryPow<float, float, float>>;
        case BinaryOpOperation_ATAN2:
            return execute<float, float, BinaryAtan2<float, float, float>>;
        case BinaryOpOperation_MOD:
            return execute<float, float, BinaryMod<float, float, float>>;
        default:
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}

MNNBinaryExecute CPUBinary::selectForInt(int type) {
    auto vecFunction = selectVector<Vec4Int, 4, int32_t>(type);
    if (nullptr != vecFunction) {
        return vecFunction;
    }
    switch (type) {
        case BinaryOpOperation_MUL:
            return execute<int32_t, int32_t, BinaryMul<int32_t, int32_t, int32_t>>;
        case BinaryOpOperation_REALDIV:
            return execute<int32_t, int32_t, BinaryRealDiv<int32_t, int32_t, int32_t>>;
        case BinaryOpOperation_FLOORDIV:
            return execute<int32_t, int32_t, BinaryFloorDiv<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_FLOORMOD:
            return execute<int32_t, int32_t, BinaryFloorMod<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_LOGICALOR:
            return execute<int32_t, int32_t, BinaryLogicalOr<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_NOTEQUAL:
            return execute<int32_t, int32_t, BinaryNotEqual<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_MOD:
            return execute<int32_t, int32_t, BinaryModInt<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_LOGICALXOR:
            return execute<int32_t, int32_t, BinaryLogicalXor<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_LEFTSHIFT:
            return execute<int32_t, int32_t, BinaryLeftShift<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_RIGHTSHIFT:
            return execute<int32_t, int32_t, BinaryRightShift<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_BITWISE_AND:
            return execute<int32_t, int32_t, BinaryBitwiseAnd<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_BITWISE_OR:
            return execute<int32_t, int32_t, BinaryBitwiseOr<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_BITWISE_XOR:
            return execute<int32_t, int32_t, BinaryBitwiseXor<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_POW:
            return execute<int32_t, int32_t, BinaryPow<int32_t, int32_t, int32_t>>;
            break;
        default:
            MNN_ERROR("Don't support binary - int compute for type %d\n", type);
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}

class CPUBinaryCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        int32_t type = op->main_as_BinaryOp()->opType();
        auto dataType = inputs[0]->getType();
        auto core = static_cast<CPUBackend*>(backend)->functions();
        auto input0Ptr = inputs[0]->host<uint8_t>();
        if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) {
            if (CPUBackend::getDataType(inputs[1]) == DataType_DT_INT8 || inputs[1]->getType().bytes() == 1) {
                if (CPUBackend::getDataType(outputs[0]) == DataType_DT_INT8 || outputs[0]->getType().bytes() == 1) {
                    auto func = CPUBinaryInt8::selectForInt8(type);
                    if (nullptr == func) {
                        return nullptr;
                    }
                    return new CPUBinaryInt8(backend, func, op->main_as_BinaryOp()->activationType());
                }
            }
        }
        if (dataType.bits == 32) {
            if (dataType.code == halide_type_int) {
                auto func = CPUBinary::selectForInt(type);
                if (nullptr == func) {
                    return nullptr;
                }
                return new CPUBinary(backend, func, op->main_as_BinaryOp()->activationType());
            } else if (dataType.code == halide_type_float) {
                auto func = core->MNNSelectBinaryFunctionForFloat(type);
                if (nullptr == func) {
                    return nullptr;
                }
                return new CPUBinary(backend, func, op->main_as_BinaryOp()->activationType());
            }
        }
        MNN_ERROR("CpuBinary: unsupported data type (bits: %d, code: %d)\n",
                  dataType.bits, dataType.code);
        return nullptr;
    }
};

REGISTER_CPU_OP_CREATOR(CPUBinaryCreator, OpType_BinaryOp);

} // namespace MNN

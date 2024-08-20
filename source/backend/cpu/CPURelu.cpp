//
//  CPURelu.cpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string.h>
#include "backend/cpu/CPURelu.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "CPUBackend.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {
CPURelu::CPURelu(Backend *b, float slope) : Execution(b) {
    auto core = static_cast<CPUBackend*>(b)->functions();
    mSlope.reset(core->bytes * core->pack);
    if (core->bytes < 4) {
        // For Lowp
        std::vector<float> tempSlope(core->pack);
        for (int i=0; i<core->pack; ++i) {
            tempSlope[i] = slope;
        }
        core->MNNFp32ToLowp(tempSlope.data(), (int16_t*)mSlope.get(), core->pack);
    } else {
        for (int i=0; i<core->pack; ++i) {
            ((float*)mSlope.get())[i] = slope;
        }
    }
}
ErrorCode CPURelu::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    mRealSize = static_cast<CPUBackend*>(backend())->getTensorSize(inputs[0]);
    if (mRealSize % core->pack != 0) {
        mCacheDst.reset(core->pack * core->bytes);
        mCacheSrc.reset(core->pack * core->bytes);
    }
    return NO_ERROR;
}

ErrorCode CPURelu::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& ib = inputs[0]->buffer();
    auto& ob = outputs[0]->buffer();

    if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) {
        const int8_t* srcO = (const int8_t*)ib.host;
        auto inInfo = TensorUtils::getQuantInfo(inputs[0]);
        auto outInfo = TensorUtils::getQuantInfo(outputs[0]);
        if (inInfo != outInfo) {
            MNN_PRINT("this relu int8 implementation has error when input output quant info mismatch\n");
        }
        int8_t zeroPoint = int8_t(outInfo[1]);
        int8_t* dstO       = (int8_t*)ob.host;
        auto size         = mRealSize;
        auto numberThread = ((CPUBackend*)backend())->threadNumber();
        int sizeQuad     = size / 16;
        int remain       = sizeQuad * 16;
        int sizeDivide = sizeQuad / numberThread;
        if (sizeQuad > 0) {
            MNN_CONCURRENCY_BEGIN(tId, numberThread) {
                int number = sizeDivide;
                if (tId == numberThread - 1) {
                    number = sizeQuad - tId * sizeDivide;
                }
                MNNReluInt8(dstO + 16 * tId * sizeDivide, srcO + 16 * tId * sizeDivide, number * 16, zeroPoint);
            }
            MNN_CONCURRENCY_END();
        }
        for (int i = remain; i < size; i++) {
            dstO[i] = srcO[i] > zeroPoint ? srcO[i] : zeroPoint;
        }
        return NO_ERROR;
    }
    auto core = static_cast<CPUBackend*>(backend())->functions();
    const uint8_t* srcO = (const uint8_t*)ib.host;
    uint8_t* dstO       = (uint8_t*)ob.host;
    auto size         = mRealSize;
    auto numberThread = ((CPUBackend*)backend())->threadNumber();
    int sizeQuad     = size / core->pack;
    int remain       = size % core->pack;
    int sizeDivide = sizeQuad / numberThread;
    if (sizeQuad > 0) {
        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            int number = sizeDivide;
            if (tId == numberThread - 1) {
                number = sizeQuad - tId * sizeDivide;
            }
            core->MNNReluWithSlopeChannel((float*)(dstO + core->pack * core->bytes * tId * sizeDivide), (const float*)(srcO + core->pack * core->bytes * tId * sizeDivide), (const float*)mSlope.get(), number, 1);
        }
        MNN_CONCURRENCY_END();
    }
    if (remain > 0) {
        ::memcpy(mCacheSrc.get(), srcO + sizeQuad * core->pack * core->bytes, remain * core->bytes);
        core->MNNReluWithSlopeChannel((float*)(mCacheDst.get()), (const float*)(mCacheSrc.get()), (const float*)mSlope.get(), 1, 1);
        ::memcpy(dstO + sizeQuad * core->pack * core->bytes, mCacheDst.get(), remain * core->bytes);
    }
    return NO_ERROR;
}

ErrorCode CPURelu6::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    mRealSize = static_cast<CPUBackend*>(backend())->getTensorSize(inputs[0]);
    if (mRealSize % core->pack != 0) {
        mCacheDst.reset(core->pack * core->bytes);
        mCacheSrc.reset(core->pack * core->bytes);
    }
    return NO_ERROR;
}

ErrorCode CPURelu6::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& ib = inputs[0]->buffer();
    auto& ob = outputs[0]->buffer();
    auto core = static_cast<CPUBackend*>(backend())->functions();
    const uint8_t* srcO = (const uint8_t*)ib.host;
    uint8_t* dstO       = (uint8_t*)ob.host;
    auto size         = mRealSize;
    auto numberThread = ((CPUBackend*)backend())->threadNumber();
    int sizeQuad     = size / core->pack;
    int remain       = size % core->pack;
    int sizeDivide = sizeQuad / numberThread;
    std::vector<uint8_t> bias(core->pack * core->bytes, 0);
    auto biasPtr = (float*)bias.data();
    if (sizeQuad > 0) {
        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            int number = sizeDivide;
            if (tId == numberThread - 1) {
                number = sizeQuad - tId * sizeDivide;
            }
            core->MNNAxByClampBroadcastUnit((float*)(dstO + core->pack * core->bytes * tId * sizeDivide), (const float*)(srcO + core->pack * core->bytes * tId * sizeDivide), biasPtr, number, 0, 0, 1, mParam.data());
        }
        MNN_CONCURRENCY_END();
    }
    if (remain > 0) {
        ::memcpy(mCacheSrc.get(), srcO + sizeQuad * core->pack * core->bytes, remain * core->bytes);
        core->MNNAxByClampBroadcastUnit((float*)(mCacheDst.get()), (const float*)(mCacheSrc.get()), biasPtr, 1, 0, 0, 1, mParam.data());
        ::memcpy(dstO + sizeQuad * core->pack * core->bytes, mCacheDst.get(), remain * core->bytes);
    }
    return NO_ERROR;
}

CPUPRelu::CPUPRelu(Backend* b, const Op* op) : MNN::Execution(b) {
    auto c = op->main_as_PRelu();
    auto core = static_cast<CPUBackend*>(b)->functions();
    mSlope.buffer().dimensions = 1;
    mSlope.buffer().dim[0].extent = UP_DIV(c->slopeCount(), core->pack) * core->pack;
    mValid = b->onAcquireBuffer(&mSlope, Backend::STATIC);
    if (!mValid) {
        return;
    }
    ::memset(mSlope.host<void>(), 0, mSlope.length(0) * core->bytes);
    if (core->bytes < 4) {
        // For Lowp
        core->MNNFp32ToLowp(c->slope()->data(), mSlope.host<int16_t>(), c->slopeCount());
    } else {
        ::memcpy(mSlope.host<void>(), c->slope()->data(), c->slopeCount() * sizeof(float));
    }
}
CPUPRelu::~CPUPRelu() {
    if (mValid) {
        backend()->onReleaseBuffer(&mSlope, Backend::STATIC);
    }
}

ErrorCode CPUPRelu::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    if (CPUBackend::getDataType(inputs[0]) == DataType_DT_INT8 || inputs[0]->getType().bytes() == 1) {
        mUseInt8 = 1;
        float inputScale = TensorUtils::getDescribe(inputs[0])->quantAttr->scale;
        float outputScale = TensorUtils::getDescribe(outputs[0])->quantAttr->scale;
        if (outputScale == 0) {
            outputScale = 0;
        } else {
            outputScale = 1.0f / outputScale;
        }
        ssize_t inputZero = static_cast<ssize_t>(TensorUtils::getDescribe(inputs[0])->quantAttr->zero);
        ssize_t outputZero = static_cast<ssize_t>(TensorUtils::getDescribe(outputs[0])->quantAttr->zero);
        ssize_t maxValue = static_cast<ssize_t>(TensorUtils::getDescribe(inputs[0])->quantAttr->max);
        ssize_t minValue = static_cast<ssize_t>(TensorUtils::getDescribe(inputs[0])->quantAttr->min);
        mQuanScalesInput.resize(1);
        mQuanScalesOutput.resize(1);
        mQuanZerosInput.resize(1);
        mQuanZerosOutput.resize(1);
        mQuanScalesInput = {inputScale};
        mQuanScalesOutput = {outputScale};
        mQuanZerosInput = {inputZero};
        mQuanZerosOutput = {outputZero};
        auto p = mSlope.host<float>();
        for (int i = 0; i < mSlope.buffer().dim[0].extent; ++i) {
            p[i] = p[i] * inputScale * outputScale;
        }
    }
    return NO_ERROR;
}

ErrorCode CPUPRelu::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto& ib            = inputs[0]->buffer();
    auto& ob            = outputs[0]->buffer();
    int sizeQuad = 1;
    for (int i=2; i<ib.dimensions; ++i) {
        sizeQuad *= ib.dim[i].extent;
    }
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto coreInt8 = static_cast<CPUBackend*>(backend())->int8Functions();
    const int channel   = ib.dim[1].extent;
    const int batch     = ib.dim[0].extent;
    int pack = 4;
    int depthQuad = UP_DIV(channel, core->pack);
    const uint8_t* srcO   = (const uint8_t*)ib.host;
    uint8_t* dstO         = (uint8_t*)ob.host;
    auto totalCount = batch * depthQuad;
    auto numberThread = ((CPUBackend*)backend())->threadNumber();
    if (mUseInt8) {
        depthQuad = UP_DIV(channel, pack);
        MNN_CONCURRENCY_BEGIN(tId, numberThread) {
            QuanPrePostParameters params;
            params.maxValue = static_cast<ssize_t>(TensorUtils::getDescribe(inputs[0])->quantAttr->max);
            params.minValue = static_cast<ssize_t>(TensorUtils::getDescribe(inputs[0])->quantAttr->min);
            params.inputScale = mQuanScalesInput.data();
            params.inputZeroPoint = mQuanZerosInput.data();
            params.outputScale = mQuanScalesOutput.data();
            params.outputZeroPoint = mQuanZerosOutput.data();
            for (int b=tId; b<totalCount; b+=numberThread) {
                auto c = b / batch;
                coreInt8->MNNReluWithSlopeChannelInt8((int8_t*)(dstO + sizeQuad * pack * b), (const int8_t*)(srcO + sizeQuad * pack * b), (const float*)(mSlope.host<uint8_t>() + core->bytes * pack * c), sizeQuad, 1, &params);
            }
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }
    MNN_CONCURRENCY_BEGIN(tId, numberThread) {
        for (int b=tId; b<totalCount; b+=numberThread) {
            auto c = b / batch;
            core->MNNReluWithSlopeChannel((float*)(dstO + sizeQuad * core->bytes * core->pack * b), (const float*)(srcO + sizeQuad * core->pack * core->bytes * b), (const float*)(mSlope.host<uint8_t>() + core->bytes * core->pack * c), sizeQuad, 1);
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

class CPUReluCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        if (op->type() == OpType_ReLU) {
            auto slope = 0.0f;
            if (nullptr != op->main() && OpParameter_Relu == op->main_type()) {
                slope = op->main_as_Relu()->slope();
            }
            return new CPURelu(backend, slope);
        }
        MNN_ASSERT(op->type() == OpType_PReLU);
        if (op->main_as_PRelu()->slopeCount() == 1) {
            return new CPURelu(backend, op->main_as_PRelu()->slope()->data()[0]);
        }
        return new CPUPRelu(backend, op);
    }
};

class CPURelu6Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        float minV = 0.0f;
        float maxV = 6.0f;
        if (nullptr != op->main()) {
            auto p = op->main_as_Relu6();
            minV = p->minValue();
            maxV = p->maxValue();
        }
        return new CPURelu6(maxV, minV, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUReluCreator, OpType_ReLU);
REGISTER_CPU_OP_CREATOR(CPUReluCreator, OpType_PReLU);
REGISTER_CPU_OP_CREATOR(CPURelu6Creator, OpType_ReLU6);
} // namespace MNN

//
//  CPUDet.cpp
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <cmath>
#include <limits>
#include "CPUDet.hpp"
#include "CPUBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/compute/CommonOptFunction.h"

namespace MNN {
ErrorCode CPUDet::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto numberThread = ((CPUBackend*)backend())->threadNumber();
    auto M = inputs[0]->length(1);
    auto core = static_cast<CPUBackend*>(backend())->functions();
    mTempMat.reset(Tensor::createDevice<float>({numberThread, M, ROUND_UP(M, core->pack)}));
    mTempRowPtrs.reset(Tensor::createDevice<float*>({numberThread, M}));
    auto success = backend()->onAcquireBuffer(mTempMat.get(), Backend::DYNAMIC);
    success &= backend()->onAcquireBuffer(mTempRowPtrs.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mTempMat.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempRowPtrs.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUDet::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto input  = inputs[0], output = outputs[0];
    auto batch = input->length(0), M = input->length(1), step = ROUND_UP(M, core->pack);
    auto computeDet = [&](int b, int tId) -> float {
#define F_IS_ZERO(v) (fabs(v) < 1e-6)
#define ADDR(row) (mTempRowPtrs->host<float*>()[tId * M + row])
#define VAL(row, col) (*(ADDR(row) + col))
        auto elimRow = [&](int row1, int row2) {
            auto ratio = -VAL(row2, row1) / VAL(row1, row1);
            float params[] = {1.f, ratio, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max()};
            int sta = row1, end = M;
            int extra = (core->pack - (end - sta) % core->pack) % core->pack;
            if (step - M >= extra) {
                end = M + extra;
            } else {
                sta -= extra - (step - M);
                end = step;
            }
            auto p1 = ADDR(row1) + sta, p2 = ADDR(row2) + sta;
            core->MNNAxByClampBroadcastUnit(p2, p2, p1, 1, core->pack, core->pack, (end - sta) / core->pack, params);
        };
        float result = 1;
        for (int i = 0; i < M; ++i) {
            auto tempPtr = mTempMat->host<float>() + (tId * M + i) * step;
            ::memcpy(tempPtr, input->host<float>() + (b * M + i) * M, M * sizeof(float));
            mTempRowPtrs->host<float*>()[tId * M + i] = tempPtr;
        }
        for (int i = 0; i < M; ++i) {
            if (F_IS_ZERO(VAL(i, i))) {
                bool swapd = false;
                for (int j = i + 1; j < M; ++j) {
                    if (!F_IS_ZERO(VAL(j, i))) {
                        std::swap(ADDR(i), ADDR(j));
                        swapd = true;
                        break;
                    }
                }
                if (!swapd) {
                    return 0;
                }
            }
            result *= VAL(i, i);
            for (int j = i + 1; j < M; ++j) {
                elimRow(i, j);
            }
        }
        return result;
    };
    
    int numberThread = ((CPUBackend*)backend())->threadNumber();
    MNN_CONCURRENCY_BEGIN(tId, numberThread) {
        for (int b = tId; b < batch; b += numberThread) {
            output->host<float>()[b] = computeDet(b, tId);
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}
class CPUDetCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUDet(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDetCreator, OpType_Det);
} // namespace MNN

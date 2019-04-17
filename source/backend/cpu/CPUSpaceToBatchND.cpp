//
//  CPUSpaceToBatchND.cpp
//  MNN
//
//  Created by MNN on 2018/12/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUSpaceToBatchND.hpp"
#include "CPUBackend.hpp"
#include "CommonOptFunction.h"
#include "Macro.h"

namespace MNN {

CPUSpaceToBatchND::CPUSpaceToBatchND(const Op* op, Backend* bn) : Execution(bn) {
    auto param        = op->main_as_SpaceBatch();
    mPadTop           = param->padding()->int32s()->data()[0];
    mPadLeft          = param->padding()->int32s()->data()[2];
    mBlockShapeHeight = param->blockShape()->int32s()->data()[0];
    mBlockShapeWidth  = param->blockShape()->int32s()->data()[1];
}

ErrorCode CPUSpaceToBatchND::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    const int channelsDiv4   = UP_DIV(input->channel(), 4);
    const int inHeight       = input->height();
    const int inWidth        = input->width();
    const int inBatch        = input->batch();
    const int outHeight      = output->height();
    const int outWidth       = output->width();
    const int outBatch       = output->batch();
    const int inPlaneC4      = inHeight * inWidth * 4;
    const int outPlaneC4     = outHeight * outWidth * 4;
    const auto inputDataBase = input->host<float>();
    auto outDataBase         = output->host<float>();
    memset(outDataBase, 0, output->size());

    for (int ob = 0; ob < outBatch; ++ob) {
        int ib      = ob % inBatch;
        int strideW = (ob / inBatch) % mBlockShapeWidth;
        int strideH = (ob / inBatch) / mBlockShapeWidth;

        auto inDataBatch  = inputDataBase + ib * channelsDiv4 * inPlaneC4;
        auto outDataBatch = outDataBase + ob * channelsDiv4 * outPlaneC4;

        const int validHStart = ALIMAX(0, (mPadTop - strideH + mBlockShapeHeight - 1) / mBlockShapeHeight);
        const int validHEnd =
            ALIMIN(outHeight, (inHeight + mPadTop - strideH + mBlockShapeHeight - 1) / mBlockShapeHeight);
        const int validWStart = ALIMAX(0, (mPadLeft - strideW + mBlockShapeWidth - 1) / mBlockShapeWidth);
        const int validWEnd =
            ALIMIN(outWidth, (inWidth + mPadLeft - strideW + mBlockShapeWidth - 1) / mBlockShapeWidth);

        for (int c = 0; c < channelsDiv4; ++c) {
            auto inDataChannel  = inDataBatch + c * inPlaneC4;
            auto outDataChannel = outDataBatch + c * outPlaneC4;

            for (int outH = validHStart; outH < validHEnd; ++outH) {
                auto outDataHeight = outDataChannel + outH * outWidth * 4;
                int inHeightIndex  = outH * mBlockShapeHeight + strideH - mPadTop;
                int inWidthIndex   = validWStart * mBlockShapeWidth + strideW - mPadLeft;
                auto inDataValid   = inDataChannel + (inHeightIndex * inWidth + inWidthIndex) * 4;
                MNNCopyC4WithStride(inDataValid, outDataHeight + validWStart * 4, mBlockShapeWidth * 4, 4,
                                    validWEnd - validWStart);
            }
        }
    }

    return NO_ERROR;
}

class SpaceBatchCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUSpaceToBatchND(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(SpaceBatchCreator, OpType_SpaceToBatchND);

} // namespace MNN

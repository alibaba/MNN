//
//  ShapeBatchToSpaceND.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SizeComputer.hpp"

namespace MNN {
class BatchToSpaceNDSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto input  = inputs[0];
        auto output = outputs[0];

        auto paramter             = op->main_as_SpaceBatch();
        const auto blockShape     = paramter->blockShape();
        const auto blockShapeData = blockShape->int32s()->data();
        int batch                 = input->batch();
        for (int i = 0; i < blockShape->dims()->data()[0]; ++i) {
            batch /= blockShapeData[i];
        }

        const auto crop     = paramter->padding();
        const auto cropData = crop->int32s()->data();
        int outputHeight    = input->height() * blockShapeData[0] - cropData[0] - cropData[1];
        int outputWidth     = input->width() * blockShapeData[1] - cropData[2] - cropData[3];

        output->buffer().dimensions = input->buffer().dimensions;
        output->setLength(0, batch);
        output->setLength(1, input->channel());
        output->setLength(2, outputHeight);
        output->setLength(3, outputWidth);
        return true;
    }
};

REGISTER_SHAPE(BatchToSpaceNDSizeComputer, OpType_BatchToSpaceND);

} // namespace MNN

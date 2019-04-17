//
//  ShapeSpaceToBatchND.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SizeComputer.hpp"

namespace MNN {
class SpaceToBatchNDSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto input  = inputs[0];
        auto output = outputs[0];

        auto paramter         = op->main_as_SpaceBatch();
        const auto blockShape = paramter->blockShape();
        int batch             = input->batch();
        for (int i = 0; i < blockShape->dims()->data()[0]; ++i) {
            batch *= blockShape->int32s()->data()[i];
        }

        const auto paddings    = paramter->padding();
        const auto paddingData = paddings->int32s()->data();
        int paddedHeight       = input->height() + paddingData[0] + paddingData[1];
        int paddedWidth        = input->width() + paddingData[2] + paddingData[3];
        int outputHeight       = paddedHeight / blockShape->int32s()->data()[0];
        int outputWidth        = paddedWidth / blockShape->int32s()->data()[1];

        output->buffer().dimensions = input->buffer().dimensions;
        output->setLength(0, batch);
        output->setLength(1, input->channel());
        output->setLength(2, outputHeight);
        output->setLength(3, outputWidth);
        return true;
    }
};

REGISTER_SHAPE(SpaceToBatchNDSizeComputer, OpType_SpaceToBatchND);
} // namespace MNN

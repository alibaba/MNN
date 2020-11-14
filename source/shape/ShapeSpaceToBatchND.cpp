//
//  ShapeSpaceToBatchND.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
class SpaceToBatchNDSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(outputs.size() == 1);
        MNN_ASSERT(inputs.size() == 1 || inputs.size() == 3);
        auto input  = inputs[0];
        auto output = outputs[0];

        int blockSize = 0;
        const int *blockData, *paddingData;
        if (inputs.size() == 3) {
            blockSize = inputs[1]->length(0);
            blockData = inputs[1]->host<int32_t>();
            paddingData = inputs[2]->host<int32_t>();
        } else {
            auto paramter         = op->main_as_SpaceBatch();
            const auto blockShape = paramter->blockShape();
            const auto paddings    = paramter->padding();
            blockSize = blockShape->dims()->data()[0];
            blockData = blockShape->int32s()->data();
            paddingData = paddings->int32s()->data();
        }
        int batch             = input->batch();
        for (int i = 0; i < blockSize; ++i) {
            batch *= blockData[i];
        }
        auto format = TensorUtils::getDescribe(input)->dimensionFormat;
        output->buffer().type = input->buffer().type;
        output->buffer().dimensions = input->buffer().dimensions;
        output->setLength(0, batch);
        TensorUtils::getDescribe(output)->dimensionFormat = format;
        if (MNN_DATA_FORMAT_NHWC != format) {
            output->setLength(1, input->length(1));
            for (int i = 0; i < blockSize; ++i) {
                int paddedLength = input->length(2+i) + paddingData[2 * i] + paddingData[2 * i+1];
                int outputLength = paddedLength / blockData[i];
                output->setLength(i+2, outputLength);
            }
        } else {
            for (int i = 0; i < blockSize; ++i) {
                int paddedLength = input->length(1 + i) + paddingData[2 * i] + paddingData[2 * i+1];
                int outputLength = paddedLength / blockData[i];
                output->setLength(i+1, outputLength);
            }
            output->setLength(1+blockSize, input->length(1+blockSize));
        }
        return true;
    }
};

REGISTER_SHAPE_INPUTS(SpaceToBatchNDSizeComputer, OpType_SpaceToBatchND, std::vector<int>({1, 2}));
} // namespace MNN

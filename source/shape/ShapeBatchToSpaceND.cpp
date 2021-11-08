//
//  ShapeBatchToSpaceND.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
namespace MNN {
class BatchToSpaceNDSizeComputer : public SizeComputer {
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
        int batch                 = input->batch();
        for (int i = 0; i < blockSize; ++i) {
            batch /= blockData[i];
        }
        output->setLength(0, batch);
        output->buffer().dimensions = input->buffer().dimensions;
        auto format = TensorUtils::getDescribe(input)->dimensionFormat;
        output->buffer().type = input->getType();
        TensorUtils::getDescribe(output)->dimensionFormat = format;
        if (MNN_DATA_FORMAT_NHWC != format) {
            output->setLength(1, input->length(1));
            for (int i = 0; i < blockSize; ++i) {
                int paddedLength = input->length(2+i) * blockData[i] - paddingData[2 * i] - paddingData[2 * i+1];
                output->setLength(i+2, paddedLength);
            }
        } else {
            output->setLength(1 + blockSize, input->length(1 + blockSize));
            for (int i = 0; i < blockSize; ++i) {
                int paddedLength = input->length(1+i) * blockData[i] - paddingData[2 * i] - paddingData[2 * i+1];
                output->setLength(i+1, paddedLength);
            }
        }
        return true;
    }
};

REGISTER_SHAPE_INPUTS(BatchToSpaceNDSizeComputer, OpType_BatchToSpaceND, std::vector<int>({1, 2}));
} // namespace MNN

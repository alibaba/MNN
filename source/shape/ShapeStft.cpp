//
//  ShapeStft.cpp
//  MNN
//
//  Created by MNN on 2024/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

class StftOpComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        int batch_size = inputs[0]->length(0);
        int signal_length = inputs[0]->length(1);
        outputs[0]->buffer().dimensions = 4;
        outputs[0]->setLength(3, 2);
        outputs[0]->setLength(0, batch_size);
        int frame_length = inputs[2]->length(0);
        int nstfts = ((signal_length - frame_length) / inputs[1]->host<int>()[0]) + 1;
        outputs[0]->setLength(1, nstfts);

        int dft_unique_bins;
        if (op->main_as_StftParam()->abs()) {
            dft_unique_bins = frame_length / 2 + 1;
        } else {
            dft_unique_bins = frame_length;
        }
        outputs[0]->setLength(2, dft_unique_bins);

        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }
};
REGISTER_SHAPE_INPUTS(StftOpComputer, OpType_Stft, std::vector<int>{1});

} // namespace MNN

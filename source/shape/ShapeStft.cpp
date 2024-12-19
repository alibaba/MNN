//
//  ShapeStft.cpp
//  MNN
//
//  Created by MNN on 2024/11/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_BUILD_AUDIO

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

class StftOpComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        int sample_length = inputs[0]->elementSize();
        auto stft = op->main_as_StftParam();
        bool abs = stft->abs();
        int n_fft = stft->n_fft();
        int hop_length = stft->hop_length();
        int frames = (sample_length - n_fft) / hop_length + 1;
        // Scalar
        outputs[0]->buffer().dimensions = 2;
        outputs[0]->setLength(0, frames);
        outputs[0]->setLength(1, n_fft / 2 + 1);
        outputs[0]->buffer().type = inputs[0]->getType();
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }
};

REGISTER_SHAPE_AUDIO(StftOpComputer, OpType_Stft);
} // namespace MNN
#endif // MNN_BUILD_AUDIO

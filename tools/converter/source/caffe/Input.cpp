//
//  Input.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"
#include "logkit.h"

class Input : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
    Input() {
    }
    virtual ~Input() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_Input;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_Input;
    }
};

void Input::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {
    MNN::InputT* input = new MNN::InputT;
    std::vector<int> dims;
    auto inputParametar = parameters.input_param();
    DCHECK(inputParametar.shape_size() == 1);
    auto shape = inputParametar.shape(0);
    for (int i = 0; i < shape.dim_size(); ++i) {
        dims.push_back(shape.dim(i));
    }
    input->dims       = dims;
    dstOp->main.value = input;
}

static OpConverterRegister<Input> a("Input");

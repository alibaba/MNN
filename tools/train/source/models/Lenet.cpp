//
//  Lenet.cpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Lenet.hpp"
#include "NN.hpp"
using namespace MNN::Express;

namespace MNN {
namespace Train {
namespace Model {

Lenet::Lenet() {
    NN::ConvOption convOption;
    convOption.kernelSize = {5, 5};
    convOption.channel    = {1, 20};
    conv1.reset(NN::Conv(convOption));
    convOption.reset();
    convOption.kernelSize = {5, 5};
    convOption.channel    = {20, 50};
    conv2.reset(NN::Conv(convOption));
    ip1.reset(NN::Linear(800, 500));
    ip2.reset(NN::Linear(500, 10));
    dropout.reset(NN::Dropout(0.5));
    registerModel({conv1, conv2, ip1, ip2, dropout});
}

std::vector<Express::VARP> Lenet::onForward(const std::vector<Express::VARP>& inputs) {
    using namespace Express;
    VARP x = inputs[0];
    x      = conv1->forward(x);
    x      = _MaxPool(x, {2, 2}, {2, 2});
    x      = conv2->forward(x);
    x      = _MaxPool(x, {2, 2}, {2, 2});
    x      = _Reshape(x, {0, -1});
    x      = _Convert(x, NCHW);
    x      = ip1->forward(x);
    x      = _Relu(x);
    x      = dropout->forward(x);
    x      = ip2->forward(x);
    x      = _Softmax(x, 1);
    return {x};
}

} // namespace Model
} // namespace Train
} // namespace MNN

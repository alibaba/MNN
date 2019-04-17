//
//  TensorTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNTestSuite.h"
#include "Tensor.hpp"

using namespace MNN;

class TensorTest : public MNNTestCase {
public:
    virtual ~TensorTest() = default;
    virtual void run() {
        // initializer
        {
            Tensor caffe(3, Tensor::CAFFE);
            assert(caffe.getDimensionType() == Tensor::CAFFE);
            assert(caffe.dimensions() == 3);
            assert(caffe.getType().bits == 32);
            assert(caffe.getType().code == halide_type_float);
            assert(caffe.host<void>() == nullptr);
            assert(caffe.deviceId() == 0);

            assert(caffe.length(0) == 0);
            assert(caffe.length(1) == 0);
            assert(caffe.length(2) == 0);
            caffe.setLength(0, 3);
            caffe.setLength(1, 5);
            caffe.setLength(2, 7);
            assert(caffe.stride(0) == 0);
            assert(caffe.stride(1) == 0);
            assert(caffe.stride(2) == 0);

            Tensor alloc(&caffe, Tensor::TENSORFLOW, true);
            assert(alloc.getDimensionType() == Tensor::TENSORFLOW);
            assert(alloc.dimensions() == 3);
            assert(alloc.getType().bits == 32);
            assert(alloc.getType().code == halide_type_float);
            assert(alloc.host<void>() != nullptr);
            assert(alloc.deviceId() == 0);

            assert(alloc.length(0) == 3);
            assert(alloc.length(1) == 5);
            assert(alloc.length(2) == 7);
            assert(alloc.stride(0) == 5 * 7);
            assert(alloc.stride(1) == 7);
            assert(alloc.stride(2) == 1);
        }
        {
            Tensor tensorflow(4, Tensor::TENSORFLOW);
            assert(tensorflow.getDimensionType() == Tensor::TENSORFLOW);
            assert(tensorflow.dimensions() == 4);
            assert(tensorflow.getType().bits == 32);
            assert(tensorflow.getType().code == halide_type_float);
            assert(tensorflow.host<void>() == nullptr);
            assert(tensorflow.deviceId() == 0);

            assert(tensorflow.length(0) == 0);
            assert(tensorflow.length(1) == 0);
            assert(tensorflow.length(2) == 0);
            assert(tensorflow.length(3) == 0);
            tensorflow.setLength(0, 3);
            tensorflow.setLength(1, 5);
            tensorflow.setLength(2, 7);
            tensorflow.setLength(3, 9);
            assert(tensorflow.stride(0) == 0);
            assert(tensorflow.stride(1) == 0);
            assert(tensorflow.stride(2) == 0);
            assert(tensorflow.stride(3) == 0);

            Tensor alloc(&tensorflow, Tensor::CAFFE_C4, true);
            assert(alloc.getDimensionType() == Tensor::CAFFE);
            assert(alloc.dimensions() == 4);
            assert(alloc.getType().bits == 32);
            assert(alloc.getType().code == halide_type_float);
            assert(alloc.host<void>() != nullptr);
            assert(alloc.deviceId() == 0);

            assert(alloc.length(0) == 3);
            assert(alloc.length(1) == 9);
            assert(alloc.length(2) == 5);
            assert(alloc.length(3) == 7);
            assert(alloc.stride(0) == (9 + 3) / 4 * 4 * 5 * 7);
            assert(alloc.stride(1) == 5 * 7);
            assert(alloc.stride(2) == 7);
            assert(alloc.stride(3) == 1);
        }

        // static creator
        {
            Tensor *tensor = Tensor::createDevice<int16_t>({1, 2, 3, 4});
            assert(tensor->getDimensionType() == Tensor::TENSORFLOW);
            assert(tensor->dimensions() == 4);
            assert(tensor->getType().bits == 16);
            assert(tensor->getType().code == halide_type_int);
            assert(tensor->host<void>() == nullptr);
            assert(tensor->deviceId() == 0);
            assert(tensor->length(0) == 1);
            assert(tensor->length(1) == 2);
            assert(tensor->length(2) == 3);
            assert(tensor->length(3) == 4);
            delete tensor;
        }
        {
            uint8_t data[] = {
                0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02,
                0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02,
            };
            Tensor *tensor = Tensor::create<uint8_t>({1, 2, 3, 4}, data, Tensor::CAFFE);
            assert(tensor->getDimensionType() == Tensor::CAFFE);
            assert(tensor->dimensions() == 4);
            assert(tensor->getType().bits == 8);
            assert(tensor->getType().code == halide_type_uint);
            assert(tensor->host<void>() != nullptr);
            assert(tensor->deviceId() == 0);
            assert(tensor->length(0) == 1);
            assert(tensor->length(1) == 2);
            assert(tensor->length(2) == 3);
            assert(tensor->length(3) == 4);
            assert(tensor->elementSize() == 1 * 2 * 3 * 4);
            for (int i = 0; i < tensor->elementSize(); i++) {
                assert(tensor->host<uint8_t>()[i] == data[i]);
            }
            delete tensor;
        }
    }
};

MNNTestSuiteRegister(TensorTest, "core/tensor");

//
//  TensorTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/Tensor.hpp>
#include "MNNTestSuite.h"

using namespace MNN;

class TensorTest : public MNNTestCase {
public:
    virtual ~TensorTest() = default;
    virtual bool run(int precision) {
        // initializer
        {
            Tensor caffe(3, Tensor::CAFFE);
            MNNTEST_ASSERT(caffe.getDimensionType() == Tensor::CAFFE);
            MNNTEST_ASSERT(caffe.dimensions() == 3);
            MNNTEST_ASSERT(caffe.getType().bits == 32);
            MNNTEST_ASSERT(caffe.getType().code == halide_type_float);
            MNNTEST_ASSERT(caffe.host<void>() == nullptr);
            MNNTEST_ASSERT(caffe.deviceId() == 0);

            MNNTEST_ASSERT(caffe.length(0) == 0);
            MNNTEST_ASSERT(caffe.length(1) == 0);
            MNNTEST_ASSERT(caffe.length(2) == 0);
            caffe.setLength(0, 3);
            caffe.setLength(1, 5);
            caffe.setLength(2, 7);
            MNNTEST_ASSERT(caffe.stride(0) == 0);
            MNNTEST_ASSERT(caffe.stride(1) == 0);
            MNNTEST_ASSERT(caffe.stride(2) == 0);

            Tensor alloc(&caffe, Tensor::TENSORFLOW, true);
            MNNTEST_ASSERT(alloc.getDimensionType() == Tensor::TENSORFLOW);
            MNNTEST_ASSERT(alloc.dimensions() == 3);
            MNNTEST_ASSERT(alloc.getType().bits == 32);
            MNNTEST_ASSERT(alloc.getType().code == halide_type_float);
            MNNTEST_ASSERT(alloc.host<void>() != nullptr);
            MNNTEST_ASSERT(alloc.deviceId() == 0);

            MNNTEST_ASSERT(alloc.length(0) == 3);
            MNNTEST_ASSERT(alloc.length(1) == 5);
            MNNTEST_ASSERT(alloc.length(2) == 7);
            MNNTEST_ASSERT(alloc.stride(0) == 5 * 7);
            MNNTEST_ASSERT(alloc.stride(1) == 7);
            MNNTEST_ASSERT(alloc.stride(2) == 1);
        }
        {
            Tensor tensorflow(4, Tensor::TENSORFLOW);
            MNNTEST_ASSERT(tensorflow.getDimensionType() == Tensor::TENSORFLOW);
            MNNTEST_ASSERT(tensorflow.dimensions() == 4);
            MNNTEST_ASSERT(tensorflow.getType().bits == 32);
            MNNTEST_ASSERT(tensorflow.getType().code == halide_type_float);
            MNNTEST_ASSERT(tensorflow.host<void>() == nullptr);
            MNNTEST_ASSERT(tensorflow.deviceId() == 0);

            MNNTEST_ASSERT(tensorflow.length(0) == 0);
            MNNTEST_ASSERT(tensorflow.length(1) == 0);
            MNNTEST_ASSERT(tensorflow.length(2) == 0);
            MNNTEST_ASSERT(tensorflow.length(3) == 0);
            tensorflow.setLength(0, 3);
            tensorflow.setLength(1, 5);
            tensorflow.setLength(2, 7);
            tensorflow.setLength(3, 9);
            MNNTEST_ASSERT(tensorflow.stride(0) == 0);
            MNNTEST_ASSERT(tensorflow.stride(1) == 0);
            MNNTEST_ASSERT(tensorflow.stride(2) == 0);
            MNNTEST_ASSERT(tensorflow.stride(3) == 0);

            Tensor alloc(&tensorflow, Tensor::CAFFE_C4, true);
            MNNTEST_ASSERT(alloc.getDimensionType() == Tensor::CAFFE);
            MNNTEST_ASSERT(alloc.dimensions() == 4);
            MNNTEST_ASSERT(alloc.getType().bits == 32);
            MNNTEST_ASSERT(alloc.getType().code == halide_type_float);
            MNNTEST_ASSERT(alloc.host<void>() != nullptr);
            MNNTEST_ASSERT(alloc.deviceId() == 0);

            MNNTEST_ASSERT(alloc.length(0) == 3);
            MNNTEST_ASSERT(alloc.length(1) == 9);
            MNNTEST_ASSERT(alloc.length(2) == 5);
            MNNTEST_ASSERT(alloc.length(3) == 7);
            MNNTEST_ASSERT(alloc.stride(0) == (9 + 3) / 4 * 4 * 5 * 7);
            MNNTEST_ASSERT(alloc.stride(1) == 5 * 7);
            MNNTEST_ASSERT(alloc.stride(2) == 7);
            MNNTEST_ASSERT(alloc.stride(3) == 1);
        }

        // static creator
        {
            Tensor *tensor = Tensor::createDevice<int16_t>({1, 2, 3, 4});
            MNNTEST_ASSERT(tensor->getDimensionType() == Tensor::TENSORFLOW);
            MNNTEST_ASSERT(tensor->dimensions() == 4);
            MNNTEST_ASSERT(tensor->getType().bits == 16);
            MNNTEST_ASSERT(tensor->getType().code == halide_type_int);
            MNNTEST_ASSERT(tensor->host<void>() == nullptr);
            MNNTEST_ASSERT(tensor->deviceId() == 0);
            MNNTEST_ASSERT(tensor->length(0) == 1);
            MNNTEST_ASSERT(tensor->length(1) == 2);
            MNNTEST_ASSERT(tensor->length(2) == 3);
            MNNTEST_ASSERT(tensor->length(3) == 4);
            delete tensor;
        }
        {
            uint8_t data[] = {
                0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02,
                0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02,
            };
            Tensor *tensor = Tensor::create<uint8_t>({1, 2, 3, 4}, data, Tensor::CAFFE);
            MNNTEST_ASSERT(tensor->getDimensionType() == Tensor::CAFFE);
            MNNTEST_ASSERT(tensor->dimensions() == 4);
            MNNTEST_ASSERT(tensor->getType().bits == 8);
            MNNTEST_ASSERT(tensor->getType().code == halide_type_uint);
            MNNTEST_ASSERT(tensor->host<void>() != nullptr);
            MNNTEST_ASSERT(tensor->deviceId() == 0);
            MNNTEST_ASSERT(tensor->length(0) == 1);
            MNNTEST_ASSERT(tensor->length(1) == 2);
            MNNTEST_ASSERT(tensor->length(2) == 3);
            MNNTEST_ASSERT(tensor->length(3) == 4);
            MNNTEST_ASSERT(tensor->elementSize() == 1 * 2 * 3 * 4);
            for (int i = 0; i < tensor->elementSize(); i++) {
                MNNTEST_ASSERT(tensor->host<uint8_t>()[i] == data[i]);
            }
            delete tensor;
        }
        return true;
    }
};

MNNTestSuiteRegister(TensorTest, "core/tensor");

//
//  TensorUtilsTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNTestSuite.h"
#include "TensorUtils.hpp"

using namespace MNN;

class TensorUtilsTest : public MNNTestCase {
public:
    virtual ~TensorUtilsTest() = default;
    virtual void run() {
        // copy
        {
            Tensor src(3, Tensor::TENSORFLOW);
            src.setLength(0, 1);
            src.setLength(1, 2);
            src.setLength(2, 3);
            src.setStride(0, 4);
            src.setStride(1, 5);
            src.setStride(2, 6);

            Tensor dst1(4, Tensor::CAFFE);
            TensorUtils::copyShape(&src, &dst1);
            assert(dst1.dimensions() == 3);
            assert(dst1.length(0) == 1);
            assert(dst1.length(1) == 2);
            assert(dst1.length(2) == 3);
            assert(dst1.stride(0) == 4);
            assert(dst1.stride(1) == 5);
            assert(dst1.stride(2) == 6);
            assert(dst1.getDimensionType() == Tensor::CAFFE);

            Tensor dst2(4, Tensor::CAFFE);
            TensorUtils::copyShape(&src, &dst2, true);
            assert(dst2.dimensions() == 3);
            assert(dst2.length(0) == 1);
            assert(dst2.length(1) == 2);
            assert(dst2.length(2) == 3);
            assert(dst2.stride(0) == 4);
            assert(dst2.stride(1) == 5);
            assert(dst2.stride(2) == 6);
            assert(dst2.getDimensionType() == Tensor::TENSORFLOW);
        }

        // layout
        {
            Tensor tensor(3, Tensor::TENSORFLOW);
            tensor.setLength(0, 1);
            tensor.setLength(1, 2);
            tensor.setLength(2, 3);
            tensor.setStride(0, 4);
            tensor.setStride(1, 5);
            tensor.setStride(2, 6);
            TensorUtils::setLinearLayout(&tensor);
            assert(tensor.stride(0) == 2 * 3);
            assert(tensor.stride(1) == 3);
            assert(tensor.stride(2) == 1);

            Tensor reorder(3, Tensor::TENSORFLOW);
            reorder.setLength(0, 1);
            reorder.setLength(1, 2);
            reorder.setLength(2, 3);
            reorder.setStride(0, 4);
            reorder.setStride(1, 5);
            reorder.setStride(2, 6);
            reorder.buffer().dim[2].flags = Tensor::REORDER_4;
            TensorUtils::setLinearLayout(&reorder);
            assert(reorder.stride(0) == 2 * ((3 + 3) / 4 * 4));
            assert(reorder.stride(1) == (3 + 3) / 4 * 4);
            assert(reorder.stride(2) == 1);
            reorder.buffer().dim[1].flags = Tensor::REORDER_8;
            TensorUtils::setLinearLayout(&reorder);
            assert(reorder.stride(0) == ((2 + 7) / 8 * 8) * ((3 + 3) / 4 * 4));
            assert(reorder.stride(1) == (3 + 3) / 4 * 4);
            assert(reorder.stride(2) == 1);
        }
    }
};
MNNTestSuiteRegister(TensorUtilsTest, "core/tensor_utils");

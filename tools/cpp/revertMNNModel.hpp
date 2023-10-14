//
//  revertMNNModel.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef REVERTMNNMODEL_HPP
#define REVERTMNNMODEL_HPP

#include "MNN_generated.h"

class Revert {
public:
    Revert(const char* originalModelFileName);
    ~Revert();
    void* getBuffer() const;
    const size_t getBufferSize() const;
    void initialize(float sparsity = 0.0f, int sparseBlockOC = 1, bool rewrite = false, bool quantizedModel = false);
    static void fillRandValue(float * data, size_t size);
    void writeExtraDescribeTensor(float* scales, float* offsets);
private:
    Revert();
    std::unique_ptr<MNN::NetT> mMNNNet;
    size_t mBufferSize;
    std::shared_ptr<uint8_t> mBuffer;
    void randStart();
    void packMNNNet();
};

#endif // REVERTMNNMODEL_HPP

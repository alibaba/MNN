//
//  SizeComputerTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNTestSuite.h"
#include "shape/SizeComputer.hpp"

using namespace MNN;

class SUTSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const {
        return true;
    }
};

class SizeComputerTest : public MNNTestCase {
public:
    virtual ~SizeComputerTest() = default;
    virtual bool run(int precision) {
        SizeComputerSuite suite;
        SUTSizeComputer* sut = new SUTSizeComputer;
        suite.insert(sut, OpType_ELU);
        MNNTEST_ASSERT(suite.search(OpType_ELU) == sut);
        return true;
    }
};
MNNTestSuiteRegister(SizeComputerTest, "core/size_computer");

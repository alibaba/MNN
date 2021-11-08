//
//  AutoStorageTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "MNNTestSuite.h"
#include "core/AutoStorage.h"

using namespace MNN;

class AutoStorageTest : public MNNTestCase {
public:
    virtual ~AutoStorageTest() = default;
    virtual bool run(int precision) {
        AutoStorage<int> storage(50);
        MNNTEST_ASSERT(storage.size() == 50);
        storage.get()[40] = 999;
        MNNTEST_ASSERT(storage.get()[40] == 999);
        storage.clear();
        MNNTEST_ASSERT(storage.get()[40] == 0);
        storage.release();
        MNNTEST_ASSERT(storage.size() == 0);
        storage.reset(100);
        MNNTEST_ASSERT(storage.size() == 100);

        auto pointer = (int *)MNNMemoryAllocAlign(50 * sizeof(int), MNN_MEMORY_ALIGN_DEFAULT);
        storage.set(pointer, 40);
        MNNTEST_ASSERT(storage.size() == 40);
        MNNTEST_ASSERT(storage.get() == pointer);
        return true;
    }
};
MNNTestSuiteRegister(AutoStorageTest, "core/auto_storage");

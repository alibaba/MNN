//
//  AutoStorageTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "AutoStorage.h"
#include "MNNTestSuite.h"

using namespace MNN;

class AutoStorageTest : public MNNTestCase {
public:
    virtual ~AutoStorageTest() = default;
    virtual void run() {
        AutoStorage<int> storage(50);
        assert(storage.size() == 50);
        storage.get()[40] = 999;
        assert(storage.get()[40] == 999);
        storage.clear();
        assert(storage.get()[40] == 0);
        storage.release();
        assert(storage.size() == 0);
        storage.reset(100);
        assert(storage.size() == 100);

        auto pointer = (int *)MNNMemoryAllocAlign(50 * sizeof(int), MNN_MEMORY_ALIGN_DEFAULT);
        storage.set(pointer, 40);
        assert(storage.size() == 40);
        assert(storage.get() == pointer);
    }
};
MNNTestSuiteRegister(AutoStorageTest, "core/auto_storage");

//
//  MemoryUtilsTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNTestSuite.h"
#include "core/MNNMemoryUtils.h"

#ifndef MNN_DEBUG_MEMORY
class MemoryUtilsTest : public MNNTestCase {
public:
    virtual ~MemoryUtilsTest() = default;
    virtual bool run(int precision) {
        {
            void *ptr = MNNMemoryAllocAlign(5, 0b111111 + 1);
            MNNTEST_ASSERT(((intptr_t)ptr & 0b111111) == 0);
            MNNMemoryFreeAlign(ptr);
        }
        {
            void *ptr = MNNMemoryCallocAlign(8 * sizeof(int), 0b111 + 1);
            MNNTEST_ASSERT(((intptr_t)ptr & 0b111) == 0);
            for (int i = 0; i < 8; i++)
                MNNTEST_ASSERT(((int *)ptr)[i] == 0);
            MNNMemoryFreeAlign(ptr);
        }
        {
            MNNTEST_ASSERT(MNNMemoryAllocAlign(0, MNN_MEMORY_ALIGN_DEFAULT) == nullptr);
            MNNTEST_ASSERT(MNNMemoryCallocAlign(0, MNN_MEMORY_ALIGN_DEFAULT) == nullptr);
            MNNTEST_ASSERT(MNNMemoryAllocAlign((size_t)-1, MNN_MEMORY_ALIGN_DEFAULT) == nullptr);
            MNNTEST_ASSERT(MNNMemoryCallocAlign((size_t)-1, MNN_MEMORY_ALIGN_DEFAULT) == nullptr);
        }
        return true;
    }
};
MNNTestSuiteRegister(MemoryUtilsTest, "core/memory_utils");
#endif

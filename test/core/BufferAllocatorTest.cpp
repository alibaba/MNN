//
//  BufferAllocatorTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNTestSuite.h"
#include "core/BufferAllocator.hpp"
#include "core/MNNMemoryUtils.h"

using namespace MNN;

class BufferAllocatorTest : public MNNTestCase {
public:
    virtual ~BufferAllocatorTest() = default;
    virtual bool run(int precision) {
        auto alignment = MNN_MEMORY_ALIGN_DEFAULT;
        BufferAllocator allocator(BufferAllocator::Allocator::createDefault());

        // alloc - free - release
        auto p1 = allocator.alloc(5);
        MNNTEST_ASSERT((size_t)p1.first % alignment == 0);
        MNNTEST_ASSERT((size_t)p1.second % alignment == 0);
        MNNTEST_ASSERT(allocator.totalSize() == 5);
        allocator.free(p1);
        MNNTEST_ASSERT(allocator.totalSize() == 5);
        allocator.release();
        MNNTEST_ASSERT(allocator.totalSize() == 0);

        // alloc separate - free - release
        auto p2 = allocator.alloc(5, true);
        MNNTEST_ASSERT((size_t)p2.first % alignment == 0);
        MNNTEST_ASSERT((size_t)p2.second % alignment == 0);
        MNNTEST_ASSERT(allocator.totalSize() == 5);
        allocator.release();
        MNNTEST_ASSERT(allocator.totalSize() == 0);

        // reuse test
        auto p3 = allocator.alloc(100);
        MNNTEST_ASSERT((size_t)p3.first % alignment == 0);
        MNNTEST_ASSERT((size_t)p3.second % alignment == 0);
        MNNTEST_ASSERT(allocator.totalSize() == 100);
        auto p4 = allocator.alloc(200);
        MNNTEST_ASSERT((size_t)p4.first % alignment == 0);
        MNNTEST_ASSERT((size_t)p4.second % alignment == 0);
        MNNTEST_ASSERT(allocator.totalSize() == 300);
        allocator.free(p4);
        auto p5 = allocator.alloc(100);
        MNNTEST_ASSERT((size_t)p5.first % alignment == 0);
        MNNTEST_ASSERT((size_t)p5.second % alignment == 0);
        MNNTEST_ASSERT(allocator.totalSize() == 300);
        MNNTEST_ASSERT(p4 == p5);
        return true;
    }
};
MNNTestSuiteRegister(BufferAllocatorTest, "core/buffer_allocator");

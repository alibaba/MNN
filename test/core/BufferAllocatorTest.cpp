//
//  BufferAllocatorTest.cpp
//  MNNTests
//
//  Created by MNN on 2019/01/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "BufferAllocator.hpp"
#include "MNNTestSuite.h"

using namespace MNN;

class BufferAllocatorTest : public MNNTestCase {
public:
    virtual ~BufferAllocatorTest() = default;
    virtual void run() {
        auto alignment = 2048;
        BufferAllocator allocator(alignment);

        // alloc - free - release
        auto p1 = allocator.alloc(5);
        assert((size_t)p1 % alignment == 0);
        assert(allocator.totalSize() == 5);
        allocator.free(p1);
        assert(allocator.totalSize() == 5);
        allocator.release();
        assert(allocator.totalSize() == 0);

        // alloc separate - free - release
        auto p2 = allocator.alloc(5, true);
        assert((size_t)p2 % alignment == 0);
        assert(allocator.totalSize() == 5);
        allocator.release();
        assert(allocator.totalSize() == 0);

        // reuse test
        auto p3 = allocator.alloc(100);
        assert((size_t)p3 % alignment == 0);
        assert(allocator.totalSize() == 100);
        auto p4 = allocator.alloc(200);
        assert((size_t)p4 % alignment == 0);
        assert(allocator.totalSize() == 300);
        allocator.free(p4);
        auto p5 = allocator.alloc(100);
        assert((size_t)p5 % alignment == 0);
        assert(allocator.totalSize() == 300);
        assert(p4 == p5);
    }
};
MNNTestSuiteRegister(BufferAllocatorTest, "core/buffer_allocator");

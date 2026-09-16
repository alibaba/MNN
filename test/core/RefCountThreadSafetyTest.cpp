//
//  RefCountThreadSafetyTest.cpp
//  MNNTests
//
//  Created by MNN on 2026/08/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//
//  Regression test for the cross-thread use-after-free crash seen on TaoLive
//  iOS. MNN::RefCount used a non-atomic counter, so releasing SharedPtr
//  references from two threads at once (a user-held VARP on one queue, Llm
//  teardown on a GCD queue) raced on the count and double-freed, surfacing
//  later as a crash in ~MetalMemRelease.
//
//  IMPORTANT — what actually has teeth where:
//    * This runtime test only detects the regression under a race detector
//      (ThreadSanitizer). In a plain optimized build it cannot: non-atomic
//      addRef/decRef is UB, and the compiler legally elides the whole racy
//      pair, so the counter never drifts and the test would pass on the buggy
//      code. Measured: the pre-fix build ran 1.6M RMW pairs in 0.019s (loop
//      removed) vs 0.9s for the fixed atomic build.
//    * The always-on guard is therefore a static_assert next to
//      RefCount::mNum in source/core/AutoStorage.h, which fails the build if
//      the counter is ever reverted to a plain int.
//
//  Run under TSAN to get real detection:
//    cmake .. -DMNN_BUILD_TEST=ON -DCMAKE_CXX_FLAGS=-fsanitize=thread \
//             -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread
//    ./run_test.out core/refcount_thread_safety

#include <thread>
#include <atomic>
#include <vector>
#include "MNNTestSuite.h"
#include "core/AutoStorage.h"

using namespace MNN;

static std::atomic<int> gLiveObjects{0};

class TestRefObj : public RefCount {
public:
    TestRefObj() {
        gLiveObjects.fetch_add(1, std::memory_order_relaxed);
    }
    virtual ~TestRefObj() {
        gLiveObjects.fetch_sub(1, std::memory_order_relaxed);
    }
};

class RefCountThreadSafetyTest : public MNNTestCase {
public:
    virtual ~RefCountThreadSafetyTest() = default;
    virtual bool run(int precision) {
        // Two owners of one object released concurrently — the exact shape of the
        // production crash. Under TSAN a non-atomic counter reports a data race
        // here; the object must be destroyed exactly once either way.
        const int kIterations = 2000;
        gLiveObjects.store(0);
        for (int i = 0; i < kIterations; ++i) {
            TestRefObj* raw = new TestRefObj;
            SharedPtr<TestRefObj> owner1(raw);
            raw->addRef();
            SharedPtr<TestRefObj> owner2(raw);

            std::thread t1([&owner1]() { owner1 = nullptr; });
            std::thread t2([&owner2]() { owner2 = nullptr; });
            t1.join();
            t2.join();
            MNNTEST_ASSERT(gLiveObjects.load() == 0);
        }

        // Concurrent balanced addRef/decRef against a live baseline reference.
        const int kThreads = 4;
        // Static storage avoids lambda capture issues on MSVC (C3493).
        static constexpr int kCycles = 20000;
        SharedPtr<TestRefObj> guard(new TestRefObj);
        TestRefObj* shared = guard.get();
        std::vector<std::thread> workers;
        for (int t = 0; t < kThreads; ++t) {
            workers.emplace_back([shared]() {
                for (int i = 0; i < kCycles; ++i) {
                    shared->addRef();
                    shared->decRef();
                }
            });
        }
        for (auto& w : workers) {
            w.join();
        }
        MNNTEST_ASSERT(shared->count() == 1);
        MNNTEST_ASSERT(gLiveObjects.load() == 1);
        guard = nullptr;
        MNNTEST_ASSERT(gLiveObjects.load() == 0);
        return true;
    }
};
MNNTestSuiteRegister(RefCountThreadSafetyTest, "core/refcount_thread_safety");

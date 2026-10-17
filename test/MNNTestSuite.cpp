//
//  MNNTestSuite.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdlib.h>
#include <map>
#include <algorithm>
#include <MNN/AutoTime.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "MNNTestSuite.h"
MNNTestSuite* MNNTestSuite::gInstance = NULL;

MNNTestSuite* MNNTestSuite::get() {
    if (gInstance == NULL)
        gInstance = new MNNTestSuite;
    return gInstance;
}

MNNTestSuite::~MNNTestSuite() {
    for (int i = 0; i < mTests.size(); ++i) {
        delete mTests[i];
    }
    mTests.clear();
}

void MNNTestSuite::add(MNNTestCase* test, const char* name) {
    test->name = name;
    mTests.push_back(test);
}

static void printTestResult(int wrong, int right, int skipped, const char* flag) {
    MNN_PRINT("TEST_NAME_UNIT%s: Unit Test %s\nTEST_CASE_AMOUNT_UNIT%s: ", flag, flag, flag);
    MNN_PRINT("{\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":%d}\n", wrong, right, skipped);
    MNN_PRINT("TEST_CASE={\"name\":\"Unit Test %s\",\"failed\":%d,\"passed\":%d}\n", flag, wrong, right);
}

// MNN: optionally skip tests whose exact name appears in the comma-separated
// MNN_TEST_SKIP env var. Used by test.sh to drop tests that hit
// device-specific upstream bugs (e.g. Mali OpenCL BUFFER-mode loop kernels)
// without losing coverage for the rest of the suite.
static bool _mnn_test_should_skip(const std::string& name) {
    static const std::vector<std::string> gSkipList = []{
        std::vector<std::string> out;
        const char* env = std::getenv("MNN_TEST_SKIP");
        if (env == nullptr || *env == '\0') return out;
        std::string s(env);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            if (comma == std::string::npos) comma = s.size();
            if (comma > pos) out.emplace_back(s.substr(pos, comma - pos));
            pos = comma + 1;
        }
        return out;
    }();
    for (auto& s : gSkipList) {
        if (name == s) return true;
    }
    return false;
}

int MNNTestSuite::run(const char* key, int precision, const char* flag) {
    if (key == NULL || strlen(key) == 0) {
        MNN_ERROR("Empty test selector; use 'all' to run the whole suite\n");
        return 1;
    }
    std::vector<std::pair<std::string, float>> runTimes;
    auto suite         = MNNTestSuite::get();
    std::string prefix = key;
    std::vector<std::string> wrongs;
    size_t runUnit = 0;
    suite->skipped = 0;
    for (int i = 0; i < suite->mTests.size(); ++i) {
        MNNTestCase* test = suite->mTests[i];
        if (test->name.find(prefix) == 0) {
            // Count every matched case before the skip check so that the
            // passed/skipped accounting below stays consistent when a case is
            // suppressed via MNN_TEST_SKIP.
            runUnit++;
            if (_mnn_test_should_skip(test->name)) {
                MNN_PRINT("\tskip %s (in MNN_TEST_SKIP)\n", test->name.c_str());
                suite->skipped++;
                continue;
            }
            MNN_PRINT("\trunning %s.\n", test->name.c_str());
            MNN::Timer _t;
            auto res = test->run(precision);
            runTimes.emplace_back(std::make_pair(test->name, _t.durationInUs() / 1000.0f));
            if (!res) {
                wrongs.emplace_back(test->name);
            }
        }
    }
    if (runUnit == 0) {
        MNN_ERROR("No test case matches selector '%s'\n", key);
        return 1;
    }
    std::sort(runTimes.begin(), runTimes.end(), [](const std::pair<std::string, float>& left, const std::pair<std::string, float>& right) {
        return left.second < right.second;
    });
    for (auto& iter : runTimes) {
        MNN_PRINT("%s cost time: %.3f ms\n", iter.first.c_str(), iter.second);
    }
    if (wrongs.empty() && runUnit > suite->skipped) {
        MNN_PRINT("√√√ all <%s> tests passed.\n", key);
    } else if (wrongs.empty()) {
        MNN_PRINT("All <%s> tests were skipped (MNN_TEST_SKIP).\n", key);
    }
    for (auto& wrong : wrongs) {
        MNN_PRINT("Error: %s\n", wrong.c_str());
    }
    printTestResult(wrongs.size(), runUnit - wrongs.size() - suite->skipped, suite->skipped, flag);
    return wrongs.size();
}

int MNNTestSuite::runAll(int precision, const char* flag) {
    auto suite = MNNTestSuite::get();
    std::vector<std::string> wrongs;
    std::vector<std::pair<std::string, float>> runTimes;
    size_t runUnit = 0;
    suite->skipped = 0;
    for (int i = 0; i < suite->mTests.size(); ++i) {
        MNNTestCase* test = suite->mTests[i];
        if (test->name.find("speed") != std::string::npos) {
            // Don't test for speed because cost
            continue;
        }
        if (test->name.find("model") != std::string::npos) {
            // Don't test for model because need resource
            continue;
        }
        runUnit++;
        if (_mnn_test_should_skip(test->name)) {
            MNN_PRINT("\tskip %s (in MNN_TEST_SKIP)\n", test->name.c_str());
            suite->skipped++;
            continue;
        }
        MNN_PRINT("\trunning %s.\n", test->name.c_str());
        MNN::Timer _t;
        auto res = test->run(precision);
        runTimes.emplace_back(std::make_pair(test->name, _t.durationInUs() / 1000.0f));
        if (!res) {
            wrongs.emplace_back(test->name);
        }
        // MNN: release per-test backend caches between cases. The OpenCL
        // backend retains image/buffer free-lists and per-kernel state that
        // can leak into a subsequent test (observed: cumprod/cumsum/ROIPooling
        // pass standalone but fail mid-sequence on Mali). gc(FULL) clears the
        // free-list pools without dropping the runtime/kernel cache.
        auto curExe = MNN::Express::ExecutorScope::Current();
        if (curExe != nullptr) {
            curExe->gc(MNN::Express::Executor::FULL);
        }
    }
    if (runUnit == 0) {
        MNN_ERROR("No test case was selected; check the test registration\n");
        return 1;
    }
    std::sort(runTimes.begin(), runTimes.end(), [](const std::pair<std::string, float>& left, const std::pair<std::string, float>& right) {
        return left.second < right.second;
    });
    for (auto& iter : runTimes) {
        MNN_PRINT("%s cost time: %.3f ms\n", iter.first.c_str(), iter.second);
    }
    if (wrongs.empty() && runUnit > suite->skipped) {
        MNN_PRINT("√√√ all tests passed.\n");
    } else if (wrongs.empty()) {
        MNN_PRINT("All tests were skipped (MNN_TEST_SKIP).\n");
    }
    for (auto& wrong : wrongs) {
        MNN_PRINT("Error: %s\n", wrong.c_str());
    }
    printTestResult(wrongs.size(), runUnit - wrongs.size() - suite->skipped, suite->skipped, flag);
    return wrongs.size();
}

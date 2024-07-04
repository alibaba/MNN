//
//  MNNTestSuite.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdlib.h>
#include <map>
#include <MNN/AutoTime.hpp>
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

static void printTestResult(int wrong, int right, const char* flag) {
    MNN_PRINT("TEST_NAME_UNIT%s: 单元测试%s\nTEST_CASE_AMOUNT_UNIT%s: ", flag, flag, flag);
    MNN_PRINT("{\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n", wrong, right);
}

int MNNTestSuite::run(const char* key, int precision, const char* flag) {
    if (key == NULL || strlen(key) == 0)
        return 0;
    std::map<std::string, float> runTimes;
    auto suite         = MNNTestSuite::get();
    std::string prefix = key;
    std::vector<std::string> wrongs;
    size_t runUnit = 0;
    for (int i = 0; i < suite->mTests.size(); ++i) {
        MNNTestCase* test = suite->mTests[i];
        if (test->name.find(prefix) == 0) {
            runUnit++;
            MNN_PRINT("\trunning %s.\n", test->name.c_str());
            MNN::Timer _t;
            auto res = test->run(precision);
            runTimes.insert(std::make_pair(test->name, _t.durationInUs() / 1000.0f));
            if (!res) {
                wrongs.emplace_back(test->name);
            }
        }
    }
    for (auto& iter : runTimes) {
        MNN_PRINT("%s cost time: %.3f ms\n", iter.first.c_str(), iter.second);
    }
    if (wrongs.empty()) {
        MNN_PRINT("√√√ all <%s> tests passed.\n", key);
    }
    for (auto& wrong : wrongs) {
        MNN_PRINT("Error: %s\n", wrong.c_str());
    }
    printTestResult(wrongs.size(), runUnit - wrongs.size(), flag);
    return wrongs.size();
}

int MNNTestSuite::runAll(int precision, const char* flag) {
    auto suite = MNNTestSuite::get();
    std::vector<std::string> wrongs;
    std::map<std::string, float> runTimes;
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
        MNN_PRINT("\trunning %s.\n", test->name.c_str());
        MNN::Timer _t;
        auto res = test->run(precision);
        runTimes.insert(std::make_pair(test->name, _t.durationInUs() / 1000.0f));
        if (!res) {
            wrongs.emplace_back(test->name);
        }
    }
    for (auto& iter : runTimes) {
        MNN_PRINT("%s cost time: %.3f ms\n", iter.first.c_str(), iter.second);
    }
    if (wrongs.empty()) {
        MNN_PRINT("√√√ all tests passed.\n");
    }
    for (auto& wrong : wrongs) {
        MNN_PRINT("Error: %s\n", wrong.c_str());
    }
    printTestResult(wrongs.size(), suite->mTests.size() - wrongs.size(), flag);
    return wrongs.size();
}

//
//  MNNTestSuite.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MNNTestSuite.h"
#include <stdlib.h>

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
    printf("TEST_NAME_UNIT%s: 单元测试%s\nTEST_CASE_AMOUNT_UNIT%s: ", flag, flag, flag);
    printf("{\"blocked\":0,\"failed\":%d,\"passed\":%d,\"skipped\":0}\n", wrong, right);
}

void MNNTestSuite::run(const char* key, int precision, const char* flag) {
    if (key == NULL || strlen(key) == 0)
        return;

    auto suite         = MNNTestSuite::get();
    std::string prefix = key;
    std::vector<std::string> wrongs;
    size_t runUnit = 0;
    for (int i = 0; i < suite->mTests.size(); ++i) {
        MNNTestCase* test = suite->mTests[i];
        if (test->name.find(prefix) == 0) {
            runUnit++;
            printf("\trunning %s.\n", test->name.c_str());
            auto res = test->run(precision);
            if (!res) {
                wrongs.emplace_back(test->name);
            }
        }
    }
    if (wrongs.empty()) {
        printf("√√√ all <%s> tests passed.\n", key);
    }
    for (auto& wrong : wrongs) {
        printf("Error: %s\n", wrong.c_str());
    }
    printTestResult(wrongs.size(), runUnit - wrongs.size(), flag);
}

void MNNTestSuite::runAll(int precision, const char* flag) {
    auto suite = MNNTestSuite::get();
    std::vector<std::string> wrongs;
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
        printf("\trunning %s.\n", test->name.c_str());
        auto res = test->run(precision);
        if (!res) {
            wrongs.emplace_back(test->name);
        }
    }
    if (wrongs.empty()) {
        printf("√√√ all tests passed.\n");
    }
    for (auto& wrong : wrongs) {
        printf("Error: %s\n", wrong.c_str());
    }
    printTestResult(wrongs.size(), suite->mTests.size() - wrongs.size(), flag);
}

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

void MNNTestSuite::run(const char* key) {
    if (key == NULL || strlen(key) == 0)
        return;

    auto suite         = MNNTestSuite::get();
    std::string prefix = key;
    for (int i = 0; i < suite->mTests.size(); ++i) {
        MNNTestCase* test = suite->mTests[i];
        if (test->name.find(prefix) == 0) {
            printf("\trunning %s.\n", test->name.c_str());
            test->run();
        }
    }
    printf("√√√ all <%s> tests passed.\n", key);
}

void MNNTestSuite::runAll() {
    auto suite = MNNTestSuite::get();
    for (int i = 0; i < suite->mTests.size(); ++i) {
        MNNTestCase* test = suite->mTests[i];
        printf("\trunning %s.\n", test->name.c_str());
        test->run();
    }
    printf("√√√ all tests passed.\n");
}

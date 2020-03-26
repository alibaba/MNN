//
//  main.cpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "MNNTestSuite.h"
#include <MNN/expr/Executor.hpp>
#include <string.h>

int main(int argc, char* argv[]) {
    if (argc > 2) {
        auto type = (MNNForwardType)atoi(argv[2]);
        FUNC_PRINT(type);
        MNN::BackendConfig config;
        config.precision = MNN::BackendConfig::Precision_High;
        MNN::Express::Executor::getGlobalExecutor()->setGlobalExecutorConfig(type, config, 1);
    }
    if (argc > 1) {
        auto name = argv[1];
        MNNTestSuite::run(name);
    } else {
        MNNTestSuite::runAll();
    }
    return 0;
}

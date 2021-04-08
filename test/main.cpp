//
//  main.cpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <MNN/expr/Executor.hpp>
#include <string.h>
#include "MNNTestSuite.h"

int main(int argc, char* argv[]) {
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        MNN_PRINT("./run_test.out [test_name] [backend] [precision]\n");
        MNN_PRINT("\t backend: 0 - CPU (default), 3 - OpenCL\n");
        MNN_PRINT("\t precision: 0 - Normal, 1 - High (default), 2 - Low\n");
        return 0;
    }
    if (argc > 2) {
        auto type = (MNNForwardType)atoi(argv[2]);
        FUNC_PRINT(type);
        MNN::BackendConfig config;
        if (argc > 3) {
            auto precision   = atoi(argv[3]);
            config.precision = (MNN::BackendConfig::PrecisionMode)precision;
        } else {
            config.precision = MNN::BackendConfig::Precision_High;
        }
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

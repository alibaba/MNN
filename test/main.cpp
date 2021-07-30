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
        MNN_PRINT("./run_test.out [test_name] [backend] [precision] [thread/mode] [flag]\n");
        MNN_PRINT("\t backend: 0 - CPU (default), 3 - OpenCL\n");
        MNN_PRINT("\t precision: 0 - Normal, 1 - High (default), 2 - Low\n");
        return 0;
    }
    int precision = (int)MNN::BackendConfig::Precision_High;
    int thread = 1;
    const char* flag = "";
    if (argc > 2) {
        auto type = (MNNForwardType)atoi(argv[2]);
        FUNC_PRINT(type);
        if (argc > 3) {
            precision   = atoi(argv[3]);
        }
        if (argc > 4) {
            thread = atoi(argv[4]);
        }
        if (argc > 5) {
            flag = argv[5];
        }
        MNN::BackendConfig config;
        config.precision = (MNN::BackendConfig::PrecisionMode)precision;
        MNN::Express::Executor::getGlobalExecutor()->setGlobalExecutorConfig(type, config, thread);
    }
    if (argc > 1) {
        auto name = argv[1];
        if (strcmp(name, "all") == 0) {
            MNNTestSuite::runAll(precision, flag);
        } else {
            MNNTestSuite::run(name, precision, flag);
        }
    } else {
        MNNTestSuite::runAll(precision, flag);
    }
    return 0;
}

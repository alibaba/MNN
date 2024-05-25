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
#include <MNN/MNNForwardType.h>
#include <MNN/expr/Executor.hpp>
#include <string.h>
#include "MNNTestSuite.h"
#include "TestUtils.h"

int main(int argc, char* argv[]) {
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        MNN_PRINT("./run_test.out [test_name] [backend] [precision] [thread/mode] [flag]\n");
        MNN_PRINT("\t backend: 0 - CPU (default), 3 - OpenCL\n");
        MNN_PRINT("\t precision: 0 - Normal, 1 - High (default), 2 - Low\n");
        return 0;
    }
    int precision = (int)MNN::BackendConfig::Precision_High;
    int memory = (int)MNN::BackendConfig::Memory_Normal;
    int thread = 1;
    const char* flag = "";
    MNN::BackendConfig config;
    config.precision = (MNN::BackendConfig::PrecisionMode)precision;
    config.memory = (MNN::BackendConfig::MemoryMode)memory;
    auto type = MNN_FORWARD_CPU;
    if (argc > 2) {
        type = (MNNForwardType)atoi(argv[2]);
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
        if (argc > 6) {
            memory = atoi(argv[6]);
        }
        FUNC_PRINT(thread);
        FUNC_PRINT(precision);
        if (precision > MNN::BackendConfig::Precision_Low_BF16) {
            MNN_ERROR("Invalid precision mode, use 0 instead\n");
            precision = 0;
        }
        FUNC_PRINT(memory);
        if (memory > MNN::BackendConfig::Memory_Low) {
            MNN_ERROR("Invalid memory mode, use 0 instead\n");
            memory = 0;
        }
        config.precision = (MNN::BackendConfig::PrecisionMode)precision;
        config.memory = (MNN::BackendConfig::MemoryMode)memory;
    }
    auto exe = MNN::Express::Executor::newExecutor(type, config, thread);
    if (exe == nullptr) {
        MNN_ERROR("Can't create executor with type:%d, exit!\n", type);
        return 0;
    }
    MNN::Express::ExecutorScope scope(exe);
    exe->setGlobalExecutorConfig(type, config, thread);
    if (argc > 1) {
        auto name = argv[1];
        if (strcmp(name, "all") == 0) {
            return MNNTestSuite::runAll(precision, flag);
        } else {
            return MNNTestSuite::run(name, precision, flag);
        }
    } else {
        return MNNTestSuite::runAll(precision, flag);
    }
    return 0;
}

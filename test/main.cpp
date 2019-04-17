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

int main(int argc, char* argv[]) {
    srand(time(NULL));
    if (argc > 1) {
        auto name = argv[1];
        MNNTestSuite::run(name);
    } else {
        MNNTestSuite::runAll();
    }
    return 0;
}

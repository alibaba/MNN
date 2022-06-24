//
//  MNNRevert2Buffer.cpp
//  MNNConverter
//
//  Created by MNN on 2021/10/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "cli.hpp"

int main(int argc, const char** argv) {
    if (argc <= 2) {
        printf("Usage: ./MNNRevert2Buffer.out XXX.json XXX.mnn\n");
        return 0;
    }
    MNN::Cli::json2mnn(argv[1], argv[2]);
    return 0;
}

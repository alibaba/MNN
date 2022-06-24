//
//  MNNDump2Json.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "cli.hpp"

int main(int argc, const char** argv) {
    if (argc <= 2) {
        printf("Usage: ./MNNDump2Json.out XXX.MNN XXX.json\n");
        return 0;
    }
    MNN::Cli::mnn2json(argv[1], argv[2], argc);
    return 0;
}

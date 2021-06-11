//
//  MNNConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cli.hpp"

int main(int argc, char *argv[]) {
    modelConfig modelPath;

    // parser command line arg
    auto res = MNN::Cli::initializeMNNConvertArgs(modelPath, argc, argv);
    if (!res) {
        return 0;
    }
    // Convert
    MNN::Cli::convertModel(modelPath);
    return 0;
}

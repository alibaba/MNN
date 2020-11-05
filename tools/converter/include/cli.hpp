//
//  cli.hpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CLI_HPP
#define CLI_HPP

#include <iostream>
#include "config.hpp"
#include "cxxopts.hpp"

class Cli {
public:
    static void printProjectBanner();
    static cxxopts::Options initializeMNNConvertArgs(modelConfig &modelPath, int argc, char **argv);
};

using namespace std;

class CommonKit {
public:
    static bool FileIsExist(string path);
};

#endif // CLI_HPP

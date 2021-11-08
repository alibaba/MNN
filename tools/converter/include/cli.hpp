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
namespace MNN {
class MNN_PUBLIC Cli {
public:
    static bool initializeMNNConvertArgs(modelConfig &modelPath, int argc, char **argv);
    static bool convertModel(modelConfig& modelPath);
};
};

class CommonKit {
public:
    static bool FileIsExist(std::string path);
};

#endif // CLI_HPP

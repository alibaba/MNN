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
    static int testconvert(const std::string& defaultCacheFile, const std::string& directName, float maxErrorRate, const std::string& configJson);
    static bool mnn2json(const char* modelFile, const char* jsonFile, int flag = 3);
    static bool json2mnn(const char* jsonFile, const char* modelFile);
};
};

#endif // CLI_HPP

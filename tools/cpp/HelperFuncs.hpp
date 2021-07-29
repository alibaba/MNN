//
//  HelperFuncs.hpp
//  MNN
//
//  Created by MNN on 2021/07/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef HELPERFUNCS_HPP
#define HELPERFUNCS_HPP

#include <string>
#include <fstream>
#include <sstream>
#include "MNN_generated.h"

namespace HelperFuncs {

std::string getModelUUID(std::string modelFile) {
    std::unique_ptr<MNN::NetT> netT;
    std::ifstream input(modelFile);
    std::ostringstream outputOs;
    outputOs << input.rdbuf();
    netT = MNN::UnPackNet(outputOs.str().c_str());
    auto net = netT.get();

    return net->mnn_uuid;
}

} // namespace HelperFuncs

#endif // HELPERFUNCS_HPP

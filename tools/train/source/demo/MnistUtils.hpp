//
//  MnistUtils.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MnistUtils_hpp
#define MnistUtils_hpp
#include "Module.hpp"
class MnistUtils {
public:
    static void train(std::shared_ptr<MNN::Train::Module> model, std::string root);
};
#endif

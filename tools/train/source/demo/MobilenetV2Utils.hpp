//
//  MobilenetV2Utils.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MobilenetV2Utils_hpp
#define MobilenetV2Utils_hpp

#include <MNN/expr/Module.hpp>
#include <string>

class MobilenetV2Utils {
public:
    static void train(std::shared_ptr<MNN::Express::Module> model, const int numClasses, const int addToLabel,
                      std::string trainImagesFolder, std::string trainImagesTxt,
                      std::string testImagesFolder, std::string testImagesTxt, const int quantBits = 8);
};

#endif

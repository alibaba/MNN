//
//  Helper.hpp
//  MNN
//
//  Created by MNN on 2019/08/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <string>
#include <set>
#include <sys/stat.h>
#include <dirent.h>
#include "Tensor.hpp"
#include "converter/source/IR/MNN_generated.h"
#include "ImageProcess.hpp"
#include "logkit.h"

#pragma once
class Helper {
public:
    static std::set<std::string> gNeedFeatureOp;

    static std::set<MNN::OpType> INT8SUPPORTED_OPS;

    static std::set<std::string> featureQuantizeMethod;
    static std::set<std::string> weightQuantizeMethod;

    static bool fileExist(const std::string& file);
    static void readImages(std::vector<std::string>& images, const std::string& filePath, const int usedImageNum);
    static void preprocessInput(MNN::CV::ImageProcess* pretreat, int targetWidth, int targetHeight,
                            const std::string& inputImageFileName, MNN::Tensor* input);
};

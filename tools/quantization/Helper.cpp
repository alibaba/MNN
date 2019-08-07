
//
//  Helper.cpp
//  MNN
//
//  Created by MNN on 2019/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "Helper.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::set<std::string> Helper::gNeedFeatureOp = {"Convolution", "ConvolutionDepthwise"};

std::set<MNN::OpType> Helper::INT8SUPPORTED_OPS = {
    MNN::OpType_ConvInt8, MNN::OpType_DepthwiseConvInt8, MNN::OpType_PoolInt8,
    // MNN::OpType_Int8ToFloat,
    // MNN::OpType_FloatToInt8,
};

std::set<std::string> Helper::featureQuantizeMethod = {"KL", "ADMM"};
std::set<std::string> Helper::weightQuantizeMethod = {"MAX_ABS", "ADMM"};


bool Helper::fileExist(const std::string& file) {
    struct stat buffer;
    return stat(file.c_str(), &buffer) == 0;
}

void Helper::readImages(std::vector<std::string>& images, const std::string& filePath, const int usedImageNum) {
    DIR* root = opendir(filePath.c_str());
    int count = 0;
    if (root == NULL) {
        MNN_ERROR("open %s failed!\n", filePath.c_str());
        return;
    }
    struct dirent* ent = readdir(root);
    while (ent != NULL) {
        if (ent->d_name[0] != '.') {
            const std::string fileName = filePath + "/" + ent->d_name;
            if (fileExist(fileName)) {
                // std::cout << "==> " << fileName << std::endl;
                // DLOG(INFO) << fileName;
                if (usedImageNum <= 0){
                    // use all images in the folder
                    images.push_back(fileName);
                    count++;
                }
                else if (count < usedImageNum) {
                    // use usedImageNum images
                    images.push_back(fileName);
                    count++;
                }
                else {
                    break;
                }
            }
        }
        ent = readdir(root);
    }
    DLOG(INFO) << "used image num: " << images.size();
}

void Helper::preprocessInput(MNN::CV::ImageProcess* pretreat, int targetWidth, int targetHeight,
                            const std::string& inputImageFileName, MNN::Tensor* input) {
    int originalWidth, originalHeight, comp;
    auto bitmap32bits = stbi_load(inputImageFileName.c_str(), &originalWidth, &originalHeight, &comp, 4);

    DCHECK(bitmap32bits != nullptr) << "input image error!";
    MNN::CV::Matrix trans;
    // choose resize or crop
    // resize method
    trans.setScale((float)(originalWidth - 1) / (float)(targetWidth - 1),
                   (float)(originalHeight - 1) / (float)(targetHeight - 1));
    // crop method
    // trans.setTranslate(16.0f, 16.0f);
    pretreat->convert(bitmap32bits, originalWidth, originalHeight, 0, input);

    stbi_image_free(bitmap32bits);
}
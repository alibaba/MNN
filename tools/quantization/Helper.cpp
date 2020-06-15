
//
//  Helper.cpp
//  MNN
//
//  Created by MNN on 2019/08/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "Helper.hpp"
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::set<std::string> Helper::gNeedFeatureOp = {"Convolution", "ConvolutionDepthwise", "Eltwise", "Pooling"};

std::set<MNN::OpType> Helper::INT8SUPPORTED_OPS = {
    MNN::OpType_ConvInt8, MNN::OpType_DepthwiseConvInt8, MNN::OpType_PoolInt8, MNN::OpType_EltwiseInt8,
    // MNN::OpType_Int8ToFloat,
    // MNN::OpType_FloatToInt8,
};

std::set<std::string> Helper::featureQuantizeMethod = {"KL", "ADMM"};
std::set<std::string> Helper::weightQuantizeMethod  = {"MAX_ABS", "ADMM"};

#if !defined(_MSC_VER)
bool Helper::fileExist(const std::string& file) {
    struct stat buffer;
    return stat(file.c_str(), &buffer) == 0;
}
#endif

void Helper::readImages(std::vector<std::string>& images, const std::string& filePath, int* usedImageNum) {
    int count = 0;
#if defined(_MSC_VER)
    WIN32_FIND_DATA ffd;
    HANDLE hFind = INVALID_HANDLE_VALUE;
    hFind = FindFirstFile(filePath.c_str(), &ffd);
    if (INVALID_HANDLE_VALUE == hFind) {
        std::cout << "open " << filePath << " failed: " << strerror(errno) << std::endl;
        return;
    }
    do {
        const std::string fileName = filePath + "\\" + ffd.cFileName;
        if(INVALID_FILE_ATTRIBUTES != GetFileAttributes(fileName.c_str()) && GetLastError() != ERROR_FILE_NOT_FOUND) {
            if (*usedImageNum == 0) {
                // use all images in the folder
                images.push_back(fileName);
                count++;
            } else if (count < *usedImageNum) {
                // use usedImageNum images
                images.push_back(fileName);
                count++;
            } else {
                break;
            }
        }
    } while (FindNextFile(hFind, &ffd) != 0);
    FindClose(hFind);
#else
    DIR* root = opendir(filePath.c_str());
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
                if (*usedImageNum == 0) {
                    // use all images in the folder
                    images.push_back(fileName);
                    count++;
                } else if (count < *usedImageNum) {
                    // use usedImageNum images
                    images.push_back(fileName);
                    count++;
                } else {
                    break;
                }
            }
        }
        ent = readdir(root);
    }
#endif
    if (*usedImageNum == 0) {
        *usedImageNum = count;
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
    pretreat->setMatrix(trans);
    pretreat->convert(bitmap32bits, originalWidth, originalHeight, 0, input);

    stbi_image_free(bitmap32bits);
}

void Helper::invertData(float* dst, const float* src, int size) {
    for (int i = 0; i < size; ++i) {
        if (src[i] == .0f) {
            dst[i] = 0.0f;
        } else {
            dst[i] = 1.0 / src[i];
        }
    }
}

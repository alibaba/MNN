//
//  ImageNoLabelDataset.cpp
//  MNN
//
//  Created by MNN on 2020/02/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

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
#include "stb_image.h"

#include "ImageNoLabelDataset.hpp"
#include "RandomGenerator.hpp"
#include <vector>
#include <string>
#include <fstream>
using namespace MNN::CV;
namespace MNN {
namespace Train {
#if !defined(_MSC_VER)
static bool _fileExist(const std::string& file) {
    struct stat buffer;
    return stat(file.c_str(), &buffer) == 0;
}
#endif
static void _readImages(std::vector<std::string>& images, const std::string& filePath) {
    int count = 0;
#if defined(_MSC_VER)
    WIN32_FIND_DATA ffd;
    HANDLE hFind = INVALID_HANDLE_VALUE;
    hFind = FindFirstFile(filePath.c_str(), &ffd);
    if (INVALID_HANDLE_VALUE == hFind) {
        MNN_ERROR("open %s failed!\n", filePath.c_str());
        return;
    }
    do {
        const std::string fileName = filePath + "\\" + ffd.cFileName;
        if(INVALID_FILE_ATTRIBUTES != GetFileAttributes(fileName.c_str()) && GetLastError() != ERROR_FILE_NOT_FOUND) {
            images.push_back(fileName);
            count++;
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
            if (_fileExist(fileName)) {
                // std::cout << "==> " << fileName << std::endl;
                // DLOG(INFO) << fileName;
                // use all images in the folder
                images.emplace_back(fileName);
                count++;
            }
        }
        ent = readdir(root);
    }
#endif
    if (images.size() == 0) {
        MNN_ERROR("Don't find any file in %s\n", filePath.c_str());
    }
}


ImageNoLabelDataset::ImageNoLabelDataset(const std::string path, const ImageDataset::ImageConfig* cfg) {
    _readImages(mFileNames, path);
    mConfig = *cfg;
    mBpp = 0;
    switch (mConfig.destFormat) {
        case CV::RGBA:
            mBpp = 4;
            break;
        case CV::RGB:
        case CV::BGR:
            mBpp = 3;
            break;
        case CV::GRAY:
            mBpp = 1;
            break;
        default:
            break;
    }
    MNN_ASSERT(mBpp > 0);
    mProcessConfig.sourceFormat = ImageFormat::RGBA;
    mProcessConfig.filterType   = MNN::CV::BILINEAR;

    for (int i = 0; i < cfg->mean.size(); i++) {
        mProcessConfig.normal[i] = cfg->scale[i];
        mProcessConfig.mean[i] = cfg->mean[i];
    }
    mProcessConfig.destFormat = cfg->destFormat;
}
Example ImageNoLabelDataset::get(size_t index) {
    MNN_ASSERT(index >= 0 && index < mFileNames.size());
    auto fileName = mFileNames[index];
    auto image = ImageDataset::convertImage(fileName, mConfig, mProcessConfig);
    Example res;
    res.first = {image};
    return res;
}
size_t ImageNoLabelDataset::size() {
    return mFileNames.size();
}
DatasetPtr ImageNoLabelDataset::create(const std::string path, const ImageDataset::ImageConfig* cfg) {
    std::shared_ptr<BatchDataset> dataset(new ImageNoLabelDataset(path, cfg));
    DatasetPtr ptr;
    ptr.mDataset = dataset;
    return ptr;
}

}
}


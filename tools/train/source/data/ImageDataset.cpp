//
//  ImageDataset.cpp
//  MNN
//
//  Created by MNN on 2019/12/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ImageDataset.hpp"
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <string>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "MNN/ImageProcess.hpp"
#include "MNN/MNNDefine.h"
#include "stb_image.h"

using namespace std;
using namespace MNN::CV;

// behave like python split
vector<string> split(const string sourceStr, string splitChar = " ") {
    vector<string> result;
    int pos   = 0;
    int start = 0;

    while ((pos = sourceStr.find(splitChar, start)) != string::npos) {
        result.emplace_back(sourceStr.substr(start, pos - start));
        start = pos + splitChar.size();
    }

    if (start < sourceStr.size()) {
        result.emplace_back(sourceStr.substr(start));
    }

    return result;
}

ImageDataset::ImageDataset(const std::string pathToImages, const std::string pathToImageTxt, ImageConfig cfg,
                           bool readAllToMemory) {
    mReadAllToMemory = readAllToMemory;
    mConfig          = cfg;

    ImageProcess::Config config;
    config.sourceFormat = ImageFormat::RGBA;
    config.filterType   = MNN::CV::BILINEAR;

    switch (cfg.destFormat) {
        case DestImageFormat::GRAY:
            config.destFormat = ImageFormat::GRAY;
            break;
        case DestImageFormat::RGB:
            config.destFormat = ImageFormat::RGB;
            break;
        case DestImageFormat::BGR:
            config.destFormat = ImageFormat::BGR;
            break;
        default:
            MNN_PRINT("not supported dest format\n");
            MNN_ASSERT(false);
            break;
    }
    mProcess.reset(ImageProcess::create(config));

    getAllDataAndLabelsFromTxt(pathToImages, pathToImageTxt);

    if (mReadAllToMemory) {
        for (int i = 0; i < mAllTxtLines.size(); i++) {
            auto dataLabelsPair = getDataAndLabelsFrom(mAllTxtLines[i]);
            mDataAndLabels.emplace_back(dataLabelsPair);
        }
    }
}

Example ImageDataset::get(size_t index) {
    if (mReadAllToMemory) {
        return {{mDataAndLabels[index].first}, {mDataAndLabels[index].second}};
    } else {
        auto dataAndLabels = getDataAndLabelsFrom(mAllTxtLines[index]);
        return {{dataAndLabels.first}, {dataAndLabels.second}};
    }
}

size_t ImageDataset::size() {
    return mAllTxtLines.size();
}

void ImageDataset::getAllDataAndLabelsFromTxt(const std::string pathToImages, std::string pathToImageTxt) {
    std::ifstream txtFile(pathToImageTxt);
    if (!txtFile.is_open()) {
        MNN_PRINT("%s: file not found\n", pathToImageTxt.c_str());
        MNN_ASSERT(false);
    }
    string line;
    while (getline(txtFile, line)) {
        vector<string> splitStr;
        splitStr = split(line, " ");
        if (splitStr.size() != 2) {
            MNN_PRINT("%s: file format error\n", pathToImageTxt.c_str());
            MNN_ASSERT(false);
        }
        std::pair<std::string, std::vector<int> > dataPair;
        dataPair.first = pathToImages + splitStr[0];
        vector<string> labels;
        labels = split(splitStr[1], ",");
        for (int i = 0; i < labels.size(); i++) {
            dataPair.second.emplace_back(atoi(labels[i].c_str()));
        }
        mAllTxtLines.emplace_back(dataPair);
    }
    txtFile.close();
}

std::pair<VARP, VARP> ImageDataset::getDataAndLabelsFrom(std::pair<std::string, std::vector<int> > dataAndLabels) {
    int originalWidth, originalHeight, comp;
    string imageName  = dataAndLabels.first;
    auto bitmap32bits = stbi_load(imageName.c_str(), &originalWidth, &originalHeight, &comp, 4);
    if (bitmap32bits == nullptr) {
        MNN_PRINT("can not open image: %s\n", imageName.c_str());
        MNN_ASSERT(false);
    }
    MNN::CV::Matrix trans;
    // choose resize or crop
    // resize method
    int oh, ow, bpp;
    if (mConfig.resizeHeight > 0 && mConfig.resizeWidth > 0) {
        trans.setScale((float)(originalWidth - 1) / (float)(mConfig.resizeWidth - 1),
                       (float)(originalHeight - 1) / (float)(mConfig.resizeHeight - 1));
        oh = mConfig.resizeHeight;
        ow = mConfig.resizeWidth;
    } else {
        trans.setScale(1.0f, 1.0f);
        oh = originalHeight;
        ow = originalWidth;
    }
    bpp = mConfig.destFormat == DestImageFormat::GRAY ? 1 : 3;
    mProcess->setMatrix(trans);

    auto data      = _Input({oh, ow, bpp}, NHWC, halide_type_of<uint8_t>());
    auto txtLabels = dataAndLabels.second;
    auto labels    = _Input({int(txtLabels.size())}, NHWC, halide_type_of<int32_t>());

    mProcess->convert(bitmap32bits, originalWidth, originalHeight, 0, data->writeMap<uint8_t>(), ow, oh, bpp, ow * bpp,
                      halide_type_of<uint8_t>());

    auto labelsDataPtr = labels->writeMap<int32_t>();
    for (int j = 0; j < txtLabels.size(); j++) {
        labelsDataPtr[j] = txtLabels[j];
    }
    stbi_image_free(bitmap32bits);

    return std::make_pair(data, labels);
}

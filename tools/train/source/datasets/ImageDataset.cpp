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
#include "MNN/ImageProcess.hpp"
#include "MNN/MNNDefine.h"
#include "stb_image.h"
#include "RandomGenerator.hpp"

using namespace std;
using namespace MNN::CV;

namespace MNN {
namespace Train {

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
DatasetPtr ImageDataset::create(const std::string pathToImages, const std::string pathToImageTxt, const ImageConfig* cfg,
                           bool readAllToMemory) {
    auto dataset = new ImageDataset;
    dataset->mReadAllToMemory = readAllToMemory;
    dataset->mConfig          = *cfg;

    dataset->mProcessConfig.sourceFormat = ImageFormat::RGBA;
    dataset->mProcessConfig.filterType   = MNN::CV::BILINEAR;

    for (int i = 0; i < cfg->mean.size(); i++) {
        dataset->mProcessConfig.normal[i] = cfg->scale[i];
        dataset->mProcessConfig.mean[i] = cfg->mean[i];
    }
    dataset->mProcessConfig.destFormat = cfg->destFormat;

    dataset->getAllDataAndLabelsFromTxt(pathToImages, pathToImageTxt);

    if (dataset->mReadAllToMemory) {
        for (int i = 0; i < dataset->mAllTxtLines.size(); i++) {
            auto dataLabelsPair = dataset->getDataAndLabelsFrom(dataset->mAllTxtLines[i]);
            dataset->mDataAndLabels.emplace_back(dataLabelsPair);
        }
    }
    DatasetPtr ptr;
    ptr.mDataset = std::shared_ptr<BatchDataset>(dataset);
    return ptr;
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

VARP ImageDataset::convertImage(const std::string& imageName, const ImageConfig& mConfig, const MNN::CV::ImageProcess::Config& mProcessConfig) {
    int originalWidth, originalHeight, comp;
    auto bitmap32bits = stbi_load(imageName.c_str(), &originalWidth, &originalHeight, &comp, 4);
    if (bitmap32bits == nullptr) {
        MNN_PRINT("can not open image: %s\n", imageName.c_str());
        MNN_ASSERT(false);
        return nullptr;
    }
    
    // choose resize or crop
    // resize method
    int oh, ow, bpp;
    if (mConfig.resizeHeight > 0 && mConfig.resizeWidth > 0) {
        oh = mConfig.resizeHeight;
        ow = mConfig.resizeWidth;
    } else {
        oh = originalHeight;
        ow = originalWidth;
    }
    bpp = 0;
    switch (mConfig.destFormat) {
        case GRAY:
            bpp = 1;
            break;
        case RGB:
        case BGR:
            bpp = 3;
            break;
        case RGBA:
        case BGRA:
            bpp = 4;
            break;
        default:
            break;
    }
    MNN_ASSERT(bpp > 0);

    std::shared_ptr<MNN::CV::ImageProcess> process;
    process.reset(ImageProcess::create(mProcessConfig));

    if (abs(mConfig.cropFraction[0] - 1.) > 1e-6 || abs(mConfig.cropFraction[1] - 1.) > 1e-6) {
        const float cropFractionH = mConfig.cropFraction[0];
        const float cropFractionW = mConfig.cropFraction[1];

        const int hCropSize = int(originalHeight * cropFractionH);
        const int wCropSize = int(originalWidth * cropFractionW);
        MNN_ASSERT(hCropSize > 0 && wCropSize > 0);
        // default center crop
        int startH = (originalHeight - hCropSize) / 2;
        int startW = (originalWidth - wCropSize) / 2;

        if (mConfig.centerOrRandomCrop == true) {
            const int maxStartPointH = originalHeight - hCropSize;
            const int maxStartPointW = originalWidth - wCropSize;
            // generate a random number between (0, maxPoint)
            auto gen = RandomGenerator::generator();
            std::uniform_int_distribution<> disH(0, maxStartPointH);
            startH = disH(gen);
            std::uniform_int_distribution<> disW(0, maxStartPointW);
            startW = disW(gen);
        }

        const int endH = startH + hCropSize;
        const int endW = startW + wCropSize;

        float srcPoints[] = {
                float(startW), float(startH),
                float(startW), float(endH - 1),
                float(endW - 1), float(startH),
                float(endW - 1), float(endH - 1),
        };
        float dstPoints[] = {
                0.0f, 0.0f,
                0.0f, float(oh - 1),
                float(ow - 1), 0.0f,
                float(ow - 1), float(oh - 1),
        };
        MNN::CV::Matrix trans;
        trans.setPolyToPoly((MNN::CV::Point*)dstPoints, (MNN::CV::Point*)srcPoints, 4);
        process->setMatrix(trans);
    } else {
        if (mConfig.resizeHeight > 0 && mConfig.resizeWidth > 0) {
            float srcPoints[] = {
                    float(0), float(0),
                    float(0), float(originalHeight - 1),
                    float(originalWidth - 1), float(0),
                    float(originalWidth - 1), float(originalHeight - 1),
            };
            float dstPoints[] = {
                    0.0f, 0.0f,
                    0.0f, float(oh - 1),
                    float(ow - 1), 0.0f,
                    float(ow - 1), float(oh - 1),
            };
            MNN::CV::Matrix trans;
            trans.setPolyToPoly((MNN::CV::Point*)dstPoints, (MNN::CV::Point*)srcPoints, 4);
            process->setMatrix(trans);
        }
    }

    auto data      = _Input({oh, ow, bpp}, NHWC, halide_type_of<float>());
    process->convert(bitmap32bits, originalWidth, originalHeight, 0, data->writeMap<float>(), ow, oh, bpp, ow * bpp,
                      halide_type_of<float>());
    stbi_image_free(bitmap32bits);
    return data;
}

std::pair<VARP, VARP> ImageDataset::getDataAndLabelsFrom(std::pair<std::string, std::vector<int> > dataAndLabels) {
    string imageName  = dataAndLabels.first;
    auto txtLabels = dataAndLabels.second;
    auto data = convertImage(imageName, mConfig, mProcessConfig);
    auto labels    = _Input({int(txtLabels.size())}, NHWC, halide_type_of<int32_t>());


    auto labelsDataPtr = labels->writeMap<int32_t>();
    for (int j = 0; j < txtLabels.size(); j++) {
        labelsDataPtr[j] = txtLabels[j];
    }

    return std::make_pair(data, labels);
}

} // namespace Train
} // namespace MNN

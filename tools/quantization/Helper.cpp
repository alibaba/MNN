
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
#include <fstream>
#include <sstream>
#include <string>
#include "core/TensorUtils.hpp"

std::set<std::string> Helper::gNotNeedFeatureOp = { "Raster", "Pooling", "ReLU", "ReLU6", "Interp", "CropAndResize", "ROIPooling", "Gather", "GatherV2", "GatherND", "ScatterNd" };

std::set<MNN::OpType> Helper::INT8SUPPORTED_OPS = {
    MNN::OpType_ConvInt8, MNN::OpType_DepthwiseConvInt8, MNN::OpType_PoolInt8, MNN::OpType_EltwiseInt8
    // MNN::OpType_Int8ToFloat,
    // MNN::OpType_FloatToInt8,
};

std::set<std::string> Helper::featureQuantizeMethod = {"EMA", "KL", "ADMM"};
std::set<std::string> Helper::weightQuantizeMethod  = {"MAX_ABS", "ADMM"};

#if !defined(_MSC_VER)
bool Helper::fileExist(const std::string& file) {
    struct stat buffer;
    return stat(file.c_str(), &buffer) == 0;
}
#endif

void Helper::readClibrationFiles(std::vector<std::string>& images, const std::string& filePath, int* usedImageNum) {
    int count = 0;
#if defined(_MSC_VER)
    WIN32_FIND_DATA ffd;
    HANDLE hFind = INVALID_HANDLE_VALUE;
    hFind = FindFirstFile((filePath + "\\*").c_str(), &ffd);
    if (INVALID_HANDLE_VALUE == hFind) {
        std::cout << "open " << filePath << " failed: " << strerror(errno) << std::endl;
        return;
    }
    
    while (FindNextFile(hFind, &ffd))
    {
        if (ffd.cFileName[0] == '.') {
            continue;
        }
        const std::string fileName = filePath + "\\" + ffd.cFileName;
        if (INVALID_FILE_ATTRIBUTES != GetFileAttributes(fileName.c_str()) && FILE_ATTRIBUTE_DIRECTORY) {
            if (*usedImageNum == 0) {
                // use all images in the folder
                images.push_back(fileName);
                count++;
            }
            else if (count < *usedImageNum) {
                // use usedImageNum images
                images.push_back(fileName);
                count++;
            }
            else {
                break;
            }
        } else {
            if (INVALID_FILE_ATTRIBUTES != GetFileAttributes(fileName.c_str()) && GetLastError() != ERROR_FILE_NOT_FOUND) {
                if (*usedImageNum == 0) {
                    // use all images in the folder
                    images.push_back(fileName);
                    count++;
                }
                else if (count < *usedImageNum) {
                    // use usedImageNum images
                    images.push_back(fileName);
                    count++;
                }
                else {
                    break;
                }
            }
        }
    }

    FindClose(hFind);
#else
    DIR* root = opendir(filePath.c_str());
    if (root == NULL) {
        MNN_ERROR("open %s failed!\n", filePath.c_str());
        return;
    }
    struct stat s;
    struct dirent* ent = readdir(root);
    while (ent != NULL) {
        if (ent->d_name[0] != '.') {
            const std::string fileName = filePath + ent->d_name;
            stat(fileName.c_str(), &s);
            if (s.st_mode & S_IFDIR) {
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
            } else {
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
        }
        ent = readdir(root);
    }
#endif

    *usedImageNum = images.size();
    DLOG(INFO) << "used dataset num: " << images.size();
}

void Helper::preprocessInput(MNN::CV::ImageProcess* pretreat, PreprocessConfig preprocessConfig, const std::string& filename, MNN::Tensor* input, InputType inputType) {
    if (inputType == InputType::IMAGE) {
        int originalWidth, originalHeight, comp;
        auto bitmap32bits = stbi_load(filename.c_str(), &originalWidth, &originalHeight, &comp, 4);

        DCHECK(bitmap32bits != nullptr) << "input image error!";

        const int hCropSize = int(originalHeight * preprocessConfig.centerCropHeight);
        const int wCropSize = int(originalWidth * preprocessConfig.centerCropWidth);
        MNN_ASSERT(hCropSize > 0 && wCropSize > 0);
        // default center crop
        int startH = (originalHeight - hCropSize) / 2;
        int startW = (originalWidth - wCropSize) / 2;

        const int endH = startH + hCropSize;
        const int endW = startW + wCropSize;

        float srcPoints[] = {
                float(startW), float(startH),
                float(startW), float(endH - 1),
                float(endW - 1), float(startH),
                float(endW - 1), float(endH - 1),
        };
        const int oh = preprocessConfig.targetHeight;
        const int ow = preprocessConfig.targetWidth;

        float dstPoints[] = {
                0.0f, 0.0f,
                0.0f, float(oh - 1),
                float(ow - 1), 0.0f,
                float(ow - 1), float(oh - 1),
        };
        MNN::CV::Matrix trans;
        trans.setPolyToPoly((MNN::CV::Point*)dstPoints, (MNN::CV::Point*)srcPoints, 4);
        
        pretreat->setMatrix(trans);
        pretreat->convert(bitmap32bits, originalWidth, originalHeight, 0, input);

        stbi_image_free(bitmap32bits);
    }
    if (inputType == InputType::SEQUENCE) {
        if (!stringEndWith(filename, ".txt")) {
            MNN_ERROR("Error: only '.txt' files are supported for sequence input.\n");
            return;
        }

        std::ifstream f(filename);
        if (!f.is_open()) {
            MNN_ERROR("open file %s failed.\n", filename.c_str());
            return;
        }

        std::string line;
        std::vector<std::vector<float> > rawData;
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            float v;
            std::vector<float> lineData;
            while (ss >> v) {
                lineData.emplace_back(v);
            }
            if (!lineData.empty()) {
                rawData.emplace_back(lineData);
            }
        }
        f.close();

        if (rawData.empty()) {
            MNN_ERROR("Error: no data found in file %s.", filename.c_str());
            return;
        }

        std::vector<float> data;
        for (int i = 0; i< rawData.size(); i++) {
            if (rawData[i].size() != rawData[0].size()) {
                MNN_ERROR("Error: sequence length not equal in input file %s\n", filename.c_str());
                return;
            }
            data.insert(data.end(), rawData[i].begin(), rawData[i].end());
        }

        std::vector<int> shape = {1, int(rawData.size()), int(rawData[0].size())};
        std::shared_ptr<MNN::Tensor> tensorWarp(MNN::Tensor::create(shape, input->getType(), data.data(), MNN::Tensor::CAFFE));
        input->copyFromHostTensor(tensorWarp.get());
    }
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

bool Helper::stringEndWith(std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

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

#include "ImageNoLabelDataset.hpp"
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
}


ImageNoLabelDataset::ImageNoLabelDataset(const std::string path, CV::ImageProcess::Config&& config, int width, int height) {
    _readImages(mFileNames, path);
    mConfig = std::move(config);
    MNN_ASSERT(mConfig.sourceFormat == RGBA);
    mWidth = width;
    mHeight = height;
    MNN_ASSERT(mWidth > 0);
    MNN_ASSERT(mHeight > 0);
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
}
Example ImageNoLabelDataset::get(size_t index) {
    MNN_ASSERT(index >= 0 && index < mFileNames.size());
    auto fileName = mFileNames[index];
    int width, height, channel;
    auto inputImage = stbi_load(fileName.c_str(), &width, &height, &channel, 4);
    Example res;
    if (nullptr == inputImage) {
        MNN_ERROR("ImageNoLabelDataset Error: Can't load %s\n", fileName.c_str());
        return res;
    }
    std::unique_ptr<ImageProcess> process(ImageProcess::create(mConfig));
    Matrix m;
    m.setScale((float)(width-1) / (float)(mWidth-1), (float)(height-1) / (float)(mHeight-1));
    process->setMatrix(m);
    auto image = _Input({mHeight, mWidth, mBpp}, NHWC, halide_type_of<float>());
    process->convert(inputImage, width, height, 0, image->writeMap<uint8_t>(), mWidth, mHeight);
    res.first = {image};
    stbi_image_free(inputImage);
    return res;
}
size_t ImageNoLabelDataset::size() {
    return mFileNames.size();
}
DatasetPtr ImageNoLabelDataset::create(const std::string path, CV::ImageProcess::Config&& config, int width, int height) {
    std::shared_ptr<BatchDataset> dataset(new ImageNoLabelDataset(path, std::move(config), width, height));
    DatasetPtr ptr;
    ptr.mDataset = dataset;
    return ptr;
}

}
}


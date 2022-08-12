//
//  pictureRotate.cpp
//  MNN
//
//  Created by MNN on 2018/09/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <vector>
#include <MNN/AutoTime.hpp>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace MNN;
using namespace MNN::CV;

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        printf("Usage: ./pictureRotate.out input.jpg angle output.jpg\n");
        return 0;
    }
    auto inputPatch = argv[1];
    auto angle      = ::atof(argv[2]);
    auto destPath   = argv[3];
    int width, height, channel;
    auto inputImage = stbi_load(inputPatch, &width, &height, &channel, 4);

    MNN_PRINT("size: %d, %d\n", width, height);
    Matrix trans;
    trans.setScale(1.0 / (width - 1), 1.0 / (height - 1));
    trans.postRotate(-angle, 0.5, 0.5);
    trans.postScale((width - 1), (height - 1));
    ImageProcess::Config config;
    config.filterType   = NEAREST;
    config.sourceFormat = RGBA;
    config.destFormat   = RGBA;
    config.wrap         = ZERO;

    std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config), ImageProcess::destroy);
    pretreat->setMatrix(trans);
    {
        std::shared_ptr<Tensor> wrapTensor(ImageProcess::createImageTensor<uint8_t>(width, height, 4, nullptr), MNN::Tensor::destroy);
        pretreat->convert((uint8_t*)inputImage, width, height, 0, wrapTensor.get());
        stbi_write_png(argv[3], width, height, 4, wrapTensor->host<uint8_t>(), 4 * width);
    }
    stbi_image_free(inputImage);

    return 0;
}

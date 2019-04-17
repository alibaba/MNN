//
//  pictureRotate.cpp
//  MNN
//
//  Created by MNN on 2018/09/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "ImageProcess.hpp"
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <vector>
#include "AutoTime.hpp"
#include "FreeImage.h"
using namespace MNN;
using namespace MNN::CV;

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        printf("Usage: ./pictureRotate.out input.jpg angle output.jpg\n");
        return 0;
    }
    auto inputPatch     = argv[1];
    auto angle          = ::atof(argv[2]);
    auto destPath       = argv[3];
    FREE_IMAGE_FORMAT f = FreeImage_GetFileType(inputPatch);
    FIBITMAP* bitmap    = FreeImage_Load(f, inputPatch);
    MNN_ASSERT(NULL != bitmap);
    auto newBitmap = FreeImage_ConvertTo32Bits(bitmap);
    FreeImage_Unload(bitmap);
    auto width  = FreeImage_GetWidth(newBitmap);
    auto height = FreeImage_GetHeight(newBitmap);
    printf("size: %d, %d\n", width, height);
    Matrix trans;
    trans.setScale(1.0 / (width - 1), 1.0 / (height - 1));
    trans.postRotate(-angle, 0.5, 0.5);
    trans.postScale((width - 1), (height - 1));
    ImageProcess::Config config;
    config.filterType   = NEAREST;
    config.sourceFormat = RGBA;
    config.destFormat   = RGBA;
    config.wrap         = ZERO;

    std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
    pretreat->setMatrix(trans);
    {
        auto rotateBitmap = FreeImage_Allocate(width, height, 32);
        std::shared_ptr<Tensor> wrapTensor(
            ImageProcess::createImageTensor<uint8_t>(width, height, 4, FreeImage_GetScanLine(rotateBitmap, 0)));
        pretreat->convert((uint8_t*)FreeImage_GetScanLine(newBitmap, 0), width, height, 0, wrapTensor.get());
        FreeImage_Save(FIF_PNG, rotateBitmap, argv[3], PNG_DEFAULT);
        FreeImage_Unload(rotateBitmap);
    }

    return 0;
}

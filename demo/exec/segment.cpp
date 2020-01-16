//
//  segment.cpp
//  MNN
//
//  Created by MNN on 2019/07/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/AutoTime.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace MNN;
using namespace MNN::CV;
using namespace MNN::Express;

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./segment.out model.mnn input.jpg output.jpg\n");
        return 0;
    }
    auto net = Variable::getInputAndOutput(Variable::loadMap(argv[1]));
    if (net.first.empty()) {
        MNN_ERROR("Invalid Model\n");
        return 0;
    }
    auto input = net.first.begin()->second;
    auto info = input->getInfo();
    if (nullptr == info) {
        MNN_ERROR("The model don't have init dim\n");
        return 0;
    }
    auto shape = input->getInfo()->dim;
    shape[0]   = 1;
    input->resize(shape);
    auto output = net.second.begin()->second;
    if (nullptr == output->getInfo()) {
        MNN_ERROR("Alloc memory or compute size error\n");
        return 0;
    }

    {
        int size_w   = 0;
        int size_h   = 0;
        int bpp      = 0;
        if (info->order == NHWC) {
            bpp = shape[3];
            size_h = shape[1];
            size_w = shape[2];
        } else {
            bpp = shape[1];
            size_h = shape[2];
            size_w = shape[3];
        }
        if (bpp == 0)
            bpp = 1;
        if (size_h == 0)
            size_h = 1;
        if (size_w == 0)
            size_w = 1;
        MNN_PRINT("input: w:%d , h:%d, bpp: %d\n", size_w, size_h, bpp);

        auto inputPatch = argv[2];
        int width, height, channel;
        auto inputImage = stbi_load(inputPatch, &width, &height, &channel, 4);
        if (nullptr == inputImage) {
            MNN_ERROR("Can't open %s\n", inputPatch);
            return 0;
        }
        MNN_PRINT("origin size: %d, %d\n", width, height);
        Matrix trans;
        // Set scale, from dst scale to src
        trans.setScale((float)(width-1) / (size_w-1), (float)(height-1) / (size_h-1));
        ImageProcess::Config config;
        config.filterType = CV::BILINEAR;
        //        float mean[3]     = {103.94f, 116.78f, 123.68f};
        //        float normals[3] = {0.017f, 0.017f, 0.017f};
        float mean[3]     = {127.5f, 127.5f, 127.5f};
        float normals[3] = {0.00785f, 0.00785f, 0.00785f};
        ::memcpy(config.mean, mean, sizeof(mean));
        ::memcpy(config.normal, normals, sizeof(normals));
        config.sourceFormat = RGBA;
        config.destFormat   = RGB;

        std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
        pretreat->setMatrix(trans);
        pretreat->convert((uint8_t*)inputImage, width, height, 0, input->writeMap<float>(), size_w, size_h, 4, 0, halide_type_of<float>());
        stbi_image_free(inputImage);
        input->unMap();
    }
    {
        //auto originOrder = output->getInfo()->order;
        output = _Convert(output, NHWC);
        //output = _Softmax(output, -1);
        auto outputInfo = output->getInfo();
        auto width = outputInfo->dim[2];
        auto height = outputInfo->dim[1];
        auto channel = outputInfo->dim[3];
        std::shared_ptr<Tensor> wrapTensor(ImageProcess::createImageTensor<uint8_t>(width, height, 4, nullptr));
        MNN_PRINT("Mask: w=%d, h=%d, c=%d\n", width, height, channel);
        auto outputHostPtr = output->readMap<float>();
        for (int y = 0; y < height; ++y) {
            auto rgbaY = wrapTensor->host<uint8_t>() + 4 * y * width;
            auto sourceY = outputHostPtr + y * width * channel;
            for (int x=0; x<width; ++x) {
                auto sourceX = sourceY + channel * x;
                int index = 0;
                float maxValue = sourceX[0];
                auto rgba = rgbaY + 4 * x;
                for (int c=1; c<channel; ++c) {
                    if (sourceX[c] > maxValue) {
                        index = c;
                        maxValue = sourceX[c];
                    }
                }
                rgba[0] = 255;
                rgba[2] = 0;
                rgba[1] = 0;
                rgba[3] = 255;
                if (15 == index) {
                    rgba[2] = 255;
                    rgba[3] = 0;
                }
            }
        }
        output->unMap();
        stbi_write_png(argv[3], width, height, 4, wrapTensor->host<uint8_t>(), 4 * width);
    }
    return 0;
}

//
//  pictureRecognition_module.cpp
//  MNN
//
//  Created by MNN on 2018/05/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <MNN/AutoTime.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace MNN::CV;

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./pictureRecognition_module.out model.mnn input0.jpg input1.jpg input2.jpg ... \n");
        return 0;
    }
    // Load module
    std::shared_ptr<MNN::Express::Module> net(MNN::Express::Module::load(std::vector<std::string>{}, std::vector<std::string>{}, argv[1]));

    // Create Input
    int batchSize = argc - 2;
    auto input = MNN::Express::_Input({batchSize, 3, 224, 224}, MNN::Express::NC4HW4);
    for (int batch = 0; batch < batchSize; ++batch) {
        int size_w   = 224;
        int size_h   = 224;
        int bpp      = 3;

        auto inputPatch = argv[batch + 2];
        int width, height, channel;
        auto inputImage = stbi_load(inputPatch, &width, &height, &channel, 4);
        if (nullptr == inputImage) {
            MNN_ERROR("Can't open %s\n", inputPatch);
            return 0;
        }
        MNN_PRINT("origin size: %d, %d\n", width, height);
        Matrix trans;
        // Set transform, from dst scale to src, the ways below are both ok
        trans.setScale((float)(width-1) / (size_w-1), (float)(height-1) / (size_h-1));
        ImageProcess::Config config;
        config.filterType = BILINEAR;
        float mean[3]     = {103.94f, 116.78f, 123.68f};
        float normals[3] = {0.017f, 0.017f, 0.017f};
        // float mean[3]     = {127.5f, 127.5f, 127.5f};
        // float normals[3] = {0.00785f, 0.00785f, 0.00785f};
        ::memcpy(config.mean, mean, sizeof(mean));
        ::memcpy(config.normal, normals, sizeof(normals));
        config.sourceFormat = RGBA;
        config.destFormat   = BGR;

        std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
        pretreat->setMatrix(trans);
        // for NC4HW4, UP_DIV(3, 4) * 4 = 4
        pretreat->convert((uint8_t*)inputImage, width, height, 0, input->writeMap<float>() + batch * 4 * 224 * 224, 224, 224, 4, 0,  halide_type_of<float>());
        stbi_image_free(inputImage);
    }
    auto outputs = net->onForward({input});
    auto output = MNN::Express::_Convert(outputs[0], MNN::Express::NHWC);
    output = MNN::Express::_Reshape(output, {0, -1});
    int topK = 10;
    auto topKV = MNN::Express::_TopKV2(output, MNN::Express::_Scalar<int>(topK));
    auto value = topKV[0]->readMap<float>();
    auto indice = topKV[1]->readMap<int>();
    for (int batch = 0; batch < batchSize; ++batch) {
        MNN_PRINT("For Input: %s \n", argv[batch+2]);
        for (int i=0; i<topK; ++i) {
            MNN_PRINT("%d, %f\n", indice[batch * topK + i], value[batch * topK + i]);
        }
    }
    return 0;
}

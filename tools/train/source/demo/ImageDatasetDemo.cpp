//
//  ImageDatasetDemo.cpp
//  MNN
//
//  Created by MNN on 2019/11/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <iostream>
#include "DataLoader.hpp"
#include "DemoUnit.hpp"
#include "ImageDataset.hpp"
#include "MNN_generated.h"

#ifdef MNN_USE_OPENCV
#include <opencv2/opencv.hpp> // use opencv to show pictures
using namespace cv;
#endif

using namespace std;

/*
 * this is an demo for how to use the ImageDataset and DataLoader
 */

class ImageDatasetDemo : public DemoUnit {
public:
    // this function is an example to use the lambda transform
    // here we use lambda transform to normalize data from 0~255 to 0~1
    static Example func(Example example) {
        // // an easier way to do this
        auto cast       = _Cast(example.data[0], halide_type_of<float>());
        example.data[0] = _Multiply(cast, _Const(1.0f / 255.0f));
        return example;
    }

    virtual int run(int argc, const char* argv[]) override {
        if (argc != 3) {
            cout << "usage: ./runTrainDemo.out ImageDatasetDemo path/to/images/ path/to/image/txt\n" << endl;

            cout << "the ImageDataset read stored images as input data.\n"
                    "use 'pathToImages' and a txt file to construct a ImageDataset.\n"
                    "the txt file should use format as below:\n"
                    "     image1.jpg label1,label2,...\n"
                    "     image2.jpg label3,label4,...\n"
                    "     ...\n"
                    "the ImageDataset would read images from:\n"
                    "     pathToImages/image1.jpg\n"
                    "     pathToImages/image2.jpg\n"
                    "     ...\n"
                 << endl;

            return 0;
        }

        std::string pathToImages   = argv[1];
        std::string pathToImageTxt = argv[2];

        // total image num
        const size_t datasetSize = 20;

        auto converImagesToFormat  = ImageDataset::DestImageFormat::BGR;
        int resizeHeight           = 224;
        int resizeWidth            = 224;
        auto config                = ImageDataset::ImageConfig(converImagesToFormat, resizeHeight, resizeWidth);
        bool readAllImagesToMemory = false;

        auto dataset = std::make_shared<ImageDataset>(pathToImages, pathToImageTxt, config, readAllImagesToMemory);

        // the lambda transform for one example, we also can do it in batch
        auto transform = std::make_shared<LambdaTransform>(func);

        // // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
        // auto transform = std::make_shared<StackTransform>();

        const int batchSize  = 1;
        const int numWorkers = 1;

        auto dataLoader = DataLoader::makeDataLoader(dataset, {transform}, batchSize, false, numWorkers);

        const size_t iterations = datasetSize / batchSize;

        for (int i = 0; i < iterations; i++) {
            auto trainData = dataLoader->next();

            auto data  = trainData[0].data[0]->readMap<float_t>();
            auto label = trainData[0].target[0]->readMap<int32_t>();

            cout << "index: " << i << " label: " << int(label[0]) << endl;

#ifdef MNN_USE_OPENCV
            // only show the first picture in the batch
            Mat image = Mat(resizeHeight, resizeWidth, CV_32FC(3), (void*)data);
            imshow("image", image);

            waitKey(-1);
#endif
        }
        // this will reset the sampler's internal state
        dataLoader->reset();
        return 0;
    }
};

DemoUnitSetRegister(ImageDatasetDemo, "ImageDatasetDemo");

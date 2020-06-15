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
#include "RandomSampler.hpp"
#include "Sampler.hpp"
#include "Transform.hpp"
#include "TransformDataset.hpp"

#ifdef MNN_USE_OPENCV
#include <opencv2/opencv.hpp> // use opencv to show pictures
using namespace cv;
#endif

using namespace std;
using namespace MNN;
using namespace MNN::Train;

/*
 * this is an demo for how to use the ImageDataset and DataLoader
 */

class ImageDatasetDemo : public DemoUnit {
public:
    // this function is an example to use the lambda transform
    // here we use lambda transform to normalize data from 0~255 to 0~1
    static Example func(Example example) {
        // // an easier way to do this
        auto cast       = _Cast(example.first[0], halide_type_of<float>());
        example.first[0] = _Multiply(cast, _Const(1.0f / 255.0f));
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

        auto converImagesToFormat  = CV::RGB;
        int resizeHeight           = 224;
        int resizeWidth            = 224;
        std::vector<float> scales = {1/255.0, 1/255.0, 1/255.0};
        std::shared_ptr<ImageDataset::ImageConfig> config(ImageDataset::ImageConfig::create(converImagesToFormat, resizeHeight, resizeWidth, scales));
        bool readAllImagesToMemory = false;
        auto dataset = ImageDataset::create(pathToImages, pathToImageTxt, config.get(), readAllImagesToMemory);

        const int batchSize  = 1;
        const int numWorkers = 1;

        auto dataLoader = dataset.createLoader(batchSize, true, false, numWorkers);

        const size_t iterations =dataLoader->iterNumber();

        for (int i = 0; i < iterations; i++) {
            auto trainData = dataLoader->next();

            auto data  = trainData[0].first[0]->readMap<float_t>();
            auto label = trainData[0].second[0]->readMap<int32_t>();

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

# 加载训练数据
该模块用于读取保存在硬盘上的数据，将其包装并输出为MNN训练可用的数据类型。该模块源码位于MNN_root/tools/train/source/data/目录下。若要使用，请包含DataLoader.hpp头文件即可，该模块中其他组件会全部导入，用于构建DataLoader。

## 相关demo
1、MNN_root/tools/train/source/demo/dataLoaderDemo.cpp
    使用MNIST数据集构建DataLoader，并进行输出显示。
2、MNN_root/tools/train/source/demo/dataLoaderTest.cpp
    使用MNIST数据集构建DataLoader，并测试DataLoader中一些组件。
3、MNN_root/tools/train/source/demo/ImageDatasetDemo.cpp
    读取硬盘上保存的图片数据，并显示出来。显示需要用到OpenCV，并在编译时打开`MNN_USE_OPENCV`宏。

## 自定义Dataset
可参考MNN_root/tools/train/source/datasets/中预置数据集的写法，继承Dataset类，实现两个抽象函数即可，例如：
```cpp
//  MnistDataset.cpp

// 返回MNIST数据集中一张图片数据，及其对应的label
Example MnistDataset::get(size_t index) {
    auto data  = _Input({1, kImageRows, kImageColumns}, NCHW, halide_type_of<uint8_t>());
    auto label = _Input({}, NCHW, halide_type_of<uint8_t>());

    auto dataPtr = mImagePtr + index * kImageRows * kImageColumns;
    ::memcpy(data->writeMap<uint8_t>(), dataPtr, kImageRows * kImageColumns);

    auto labelPtr = mLabelsPtr + index;
    ::memcpy(label->writeMap<uint8_t>(), labelPtr, 1);

    auto returnIndex = _Const(index);
    // return the index for test
    return {{data, returnIndex}, {label}};
}
// 返回数据集大小，对于MNIST训练集是60000，测试集是10000
size_t MnistDataset::size() {
    return mImages->getInfo()->dim[0];
}
```

## DataLoader使用示例
使用流程：自定义Dataset，构造DataLoader，读取数据，DataLoader->reset();
```cpp
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

            // ImageDataset的数据格式，采用的是ImageNet数据集的格式，你也可以写一个自己的数据集，自定义格式
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

        // ImageDataset可配置数据预处理
        auto converImagesToFormat  = ImageDataset::DestImageFormat::RGB;
        int resizeHeight           = 224;
        int resizeWidth            = 224;
        std::vector<float> scales = {1/255.0, 1/255.0, 1/255.0};
        auto config                = ImageDataset::ImageConfig(converImagesToFormat, resizeHeight, resizeWidth, scales);
        bool readAllImagesToMemory = false;

        // 构建ImageDataset
        auto dataset = std::make_shared<ImageDataset>(pathToImages, pathToImageTxt, config, readAllImagesToMemory);

        const int batchSize  = 1;
        const int numWorkers = 1;

        // 构建DataLoader，这里会将一个batch数据stack为一个VARP(Tensor)
        auto dataLoader = std::shared_ptr<DataLoader>(DataLoader::makeDataLoader(dataset, batchSize, true, false, numWorkers));

        const size_t iterations = dataset->size() / batchSize;

        for (int i = 0; i < iterations; i++) {
            // 读取数据
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
        
        // 每完整过一次数据集必须重置DataLoader
        // this will reset the sampler's internal state
        dataLoader->reset();
        return 0;
    }
};

DemoUnitSetRegister(ImageDatasetDemo, "ImageDatasetDemo");
```

## 相关类和概念
### VARP
MNN动态图中的变量，类似于pytorch中的Tensor

### Example
DataLoader输出数据的最小单位

```cpp
/**
 First: data: a vector of input tensors (for single input dataset is only one)
 Second: target: a vector of output tensors (for single output dataset is only one)
 */
typedef std::pair<std::vector<VARP>, std::vector<VARP>> Example;
```
可以看到一个Example是一个数据对，其first部分是输入，second部分是target。由于网络有可能有多个输入和多个target，所以first和second都是vector结构。

### RandomSampler : public Sampler
随机采样序列生成器，例如图片数据集中有1000张图片，则生成采样序列0~999，根据配置指定是否进行shuffle
```cpp
public:
	// size: 采样序列长度
	// shuffle: 是否生成随机采样序列
    explicit RandomSampler(size_t size, bool shuffle = true);
	// 重置采样器内部状态
    void reset(size_t size) override;
	// 采样器长度
    size_t size() override;
	// 返回内部生成的采样序列
    const std::vector<size_t> indices();
	// 返回已经使用的采样序列数量
    size_t index();
	// 获取下一个长度为batchSize的采样序列
    std::vector<size_t> next(size_t batchSize) override;

private:
    std::vector<size_t> mIndices;
    size_t mIndex = 0;
    bool mShuffle;
```

### Dataset
数据集抽象基类，用户自定义数据集需继承此基类，并实现抽象函数，可参考MNN_root/tools/train/source/datasets/中预置数据集的写法
```cpp
// 返回数据集的大小，例如1000张图片的数据集，其大小为1000
virtual size_t size() = 0;
// 返回数据集中指定index的数据，如给定123，返回第123张图片数据
virtual Example get(size_t index) = 0;
// 返回数据集中指定index的一批数据，为一个batch
std::vector<Example> getBatch(std::vector<size_t> indices);
```

### Transform
抽象基类，对一个batch中的每一个数据进行某个变换，可以是一些预处理等

### BatchTransform
抽象基类，对一个batch的数据进行某个变换，可以是一些预处理等

### StackTransform : public BatchTransform
将一个Dataset输出的vector<Example>表示的一个batch数据合成一个VARP，即
Stack( (c, h, w), (c, h, w), (c, h, w)... ) --> (n, c, h, w)

### LambdaTransform : public Transform
对Dataset输出的每一个Example进行单独处理，例如中心化，归一化等

### TransformDataset : public Dataset
对Dataset进行某种Transform，仍是一个Dataset，用于输出数据

### DataLoaderConfig
对DataLoader进行配置，可配置项为：
> batchSize: 指定batch大小
> numWorkers: 多线程预读取的线程数

### DataLoader
根据采样器生成的采样序列，到对应的Dataset中取得对应的数据并输出
```cpp
// 构造函数
DataLoader(std::shared_ptr<BatchDataset> dataset, std::shared_ptr<Sampler> sampler,
               std::shared_ptr<DataLoaderConfig> config);
// 构造DataLoader，无Transform
static DataLoader* makeDataLoader(std::shared_ptr<BatchDataset> dataset,
                                  const int batchSize,
                                  const bool stack = true, // 是否将一个batch数据叠加为一个VARP(Tensor)
                                  const bool shuffle = true,
                                  const int numWorkers = 0);
// 构造DataLoader，有Transform，Transform可多个叠加
static DataLoader* makeDataLoader(std::shared_ptr<BatchDataset> dataset,
                                  std::vector<std::shared_ptr<BatchTransform>> transforms,
                                  const int batchSize,
                                  const bool shuffle = true,
                                  const int numWorkers = 0);
// 指定batch size后，迭代多少次用完全部数据，最后一个batch不足batchsize也会输出
size_t iterNumber() const;
// 数据集大小
size_t size() const;
// 输出一个batch的数据
std::vector<Example> next();
// 清空内部数据队列，重置内部采样器
void clean();
// clean()，并重新预读取，Dataset每次数据全部输出完毕，必须reset
void reset();
```
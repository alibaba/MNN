//
//  ImageDataset.hpp
//  MNN
//
//  Created by MNN on 2019/12/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ImageDataset_hpp
#define ImageDataset_hpp

#include <string>
#include <utility>
#include <vector>
#include "train/source/data/Dataset.hpp"
#include "train/source/data/Example.hpp"
#include <MNN/ImageProcess.hpp>

//
// the ImageDataset read stored images as input data.
// use 'pathToImages' and a txt file to construct a ImageDataset.
// the txt file should use format as below:
//      image1.jpg label1,label2,...
//      image2.jpg label3,label4,...
//      ...
// the ImageDataset would read images from:
//      pathToImages/image1.jpg
//      pathToImages/image2.jpg
//      ...
//

namespace MNN {
namespace Train {
class MNN_PUBLIC ImageDataset : public Dataset {
public:
    class ImageConfig {
    public:
        static ImageConfig* create(CV::ImageFormat destFmt = CV::GRAY, int resizeH = 0, int resizeW = 0,
                    std::vector<float> s = {1, 1, 1, 1}, std::vector<float> m = {0, 0, 0, 0},
                    std::vector<float> cropFract = {1/*height*/, 1/*width*/}, const bool centerOrRandom = false/*false:center*/) {
            auto config = new ImageConfig;
            config->destFormat   = destFmt;
            config->resizeHeight = resizeH;
            config->resizeWidth  = resizeW;
            config->scale = s;
            config->mean = m;
            MNN_ASSERT(cropFract.size() == 2);
            MNN_ASSERT(cropFract[0] > 0 && cropFract[0] <= 1);
            MNN_ASSERT(cropFract[1] > 0 && cropFract[1] <= 1);
            config->cropFraction = cropFract;
            config->centerOrRandomCrop = centerOrRandom;
            return config;
        }
        CV::ImageFormat destFormat;
        int resizeHeight;
        int resizeWidth;
        std::vector<float> scale;
        std::vector<float> mean;
        std::vector<float> cropFraction;
        bool centerOrRandomCrop;
    };

    static DatasetPtr create(const std::string pathToImages, const std::string pathToImageTxt,
                          const ImageConfig* cfg, bool readAllToMemory = false);
    static Express::VARP convertImage(const std::string& imageName, const ImageConfig& config, const MNN::CV::ImageProcess::Config& cvConfig);

    Example get(size_t index) override;

    size_t size() override;

private:
    ImageDataset(){}
    bool mReadAllToMemory;
    std::vector<std::pair<std::string, std::vector<int> > > mAllTxtLines;
    std::vector<std::pair<VARP, VARP> > mDataAndLabels;
    ImageConfig mConfig;
    MNN::CV::ImageProcess::Config mProcessConfig;

    void getAllDataAndLabelsFromTxt(const std::string pathToImages, std::string pathToImageTxt);
    std::pair<VARP, VARP> getDataAndLabelsFrom(std::pair<std::string, std::vector<int> > dataAndLabels);
};
} // namespace Train
} // namespace MNN

#endif // ImageDataset_hpp

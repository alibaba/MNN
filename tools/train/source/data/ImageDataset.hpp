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
#include "Dataset.hpp"
#include "Example.hpp"
#include "MNN/ImageProcess.hpp"

using namespace MNN;
using namespace MNN::Train;

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
class MNN_PUBLIC ImageDataset : public Dataset {
public:
    enum DestImageFormat {
        GRAY,
        RGB,
        BGR,
    };

    struct ImageConfig {
        ImageConfig(DestImageFormat destFmt = DestImageFormat::GRAY, int resizeH = 0, int resizeW = 0) {
            destFormat   = destFmt;
            resizeHeight = resizeH;
            resizeWidth  = resizeW;
        }
        DestImageFormat destFormat;
        int resizeHeight;
        int resizeWidth;
    };

    explicit ImageDataset(const std::string pathToImages, const std::string pathToImageTxt,
                          ImageConfig cfg = ImageConfig(), bool readAllToMemory = false);

    Example get(size_t index) override;

    size_t size() override;

private:
    bool mReadAllToMemory;
    std::vector<std::pair<std::string, std::vector<int> > > mAllTxtLines;
    std::vector<std::pair<VARP, VARP> > mDataAndLabels;
    std::shared_ptr<MNN::CV::ImageProcess> mProcess = nullptr;
    ImageConfig mConfig;

    void getAllDataAndLabelsFromTxt(const std::string pathToImages, std::string pathToImageTxt);
    std::pair<VARP, VARP> getDataAndLabelsFrom(std::pair<std::string, std::vector<int> > dataAndLabels);
};

#endif // ImageDataset_hpp

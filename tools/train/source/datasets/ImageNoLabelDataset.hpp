//
//  ImageNoLabelDataset.hpp
//  MNN
//
//  Created by MNN on 2020/02/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ImageNoLabelDataset_hpp
#define ImageNoLabelDataset_hpp
#include <MNN/expr/Expr.hpp>
#include <MNN/ImageProcess.hpp>
#include "ImageDataset.hpp"
namespace MNN {
namespace Train {
class MNN_PUBLIC ImageNoLabelDataset : public Dataset {
public:
    Example get(size_t index) override;
    size_t size() override;
    const std::vector<std::string>& files() const {
        return mFileNames;
    }
    static DatasetPtr create(const std::string path, const ImageDataset::ImageConfig* cfg);
private:
    explicit ImageNoLabelDataset(const std::string path, const ImageDataset::ImageConfig* cfg);
    std::vector<std::string> mFileNames;
    ImageDataset::ImageConfig mConfig;
    CV::ImageProcess::Config mProcessConfig;

    int mBpp;
};
}
}
#endif

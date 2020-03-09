#ifndef ImageNoLabelDataset_hpp
#define ImageNoLabelDataset_hpp
#include <MNN/expr/Expr.hpp>
#include <MNN/ImageProcess.hpp>
#include "Dataset.hpp"
namespace MNN {
namespace Train {
class MNN_PUBLIC ImageNoLabelDataset : public Dataset {
public:
    Example get(size_t index) override;
    size_t size() override;
    const std::vector<std::string>& files() const {
        return mFileNames;
    }
    static DatasetPtr create(const std::string path, CV::ImageProcess::Config&& config, int width, int height);
private:
    explicit ImageNoLabelDataset(const std::string path, CV::ImageProcess::Config&& config, int width, int height);
    std::vector<std::string> mFileNames;
    CV::ImageProcess::Config mConfig;
    int mWidth;
    int mHeight;
    int mBpp;
};
}
}
#endif

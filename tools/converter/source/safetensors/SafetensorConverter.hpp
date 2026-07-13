#ifndef SafetensorConverter_hpp
#define SafetensorConverter_hpp
#include <string>
#include <vector>
#include <MNN/MNNDefine.h>
#include <MNN/expr/Expr.hpp>
#include "config.hpp"
namespace MNN {
namespace SafeTensors {
class MNN_PUBLIC Converter {
public:
    Converter(const std::string& jsonFile);
    ~ Converter();
    std::vector<std::string> listModels() const;
    void loadSafeTensors(const std::string& safeTensorFile);
    bool convert(const std::string& name, modelConfig& modelPath);
    MNN::Express::VARP loadTensor(const std::string& name, bool printNotFound = true) const;
    bool hasTensor(const std::string& name) const;
    struct Content;
private:
    Content* mMain = nullptr;
};
};
};

#endif


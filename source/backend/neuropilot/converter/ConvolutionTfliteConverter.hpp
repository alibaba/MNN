#ifndef ConvolutionTfliteConverter_hpp
#define ConvolutionTfliteConverter_hpp
#include "ConvertTflite.hpp"
namespace MNN {
class ConvolutionTfliteConverter : public ConvertTflite::Convert {
public:
    virtual ConvertTflite::CommandBuffer onExecute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, ConvertTflite* root) override;

};
};

#endif

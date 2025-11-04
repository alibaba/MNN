#ifndef MTKEXT_hpp
#define MTKEXT_hpp
#include "ConvertTflite.hpp"
namespace MNN {
class MTKEXT : public ConvertTflite::Convert {
public:
    virtual ConvertTflite::CommandBuffer onExecute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, ConvertTflite* root) override;

};
};



#endif

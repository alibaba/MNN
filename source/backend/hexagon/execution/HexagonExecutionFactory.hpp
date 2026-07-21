#ifndef HexagonExecutionFactory_hpp
#define HexagonExecutionFactory_hpp
#include "core/Execution.hpp"
namespace MNN {
class HexagonExecutionFactory {
public:
    static Execution* create(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, Backend* backend);
};
};


#endif

#ifndef HexagonRelu_hpp
#define HexagonRelu_hpp

#include "HexagonExecution.hpp"

namespace MNN {

class HexagonRelu : public HexagonExecution {
public:
    static HexagonRelu* create(Backend* backend, const Op* op);
    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    HexagonRelu(Backend* backend, float slope);
    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    float mSlope = 0.0f;
};

} // namespace MNN

#endif

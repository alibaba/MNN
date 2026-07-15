#ifndef HexagonRelu6_hpp
#define HexagonRelu6_hpp

#include "HexagonExecution.hpp"

namespace MNN {

class HexagonRelu6 : public HexagonExecution {
public:
    static HexagonRelu6* create(Backend* backend, const Op* op);
    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    HexagonRelu6(Backend* backend, float minValue, float maxValue);
    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    float mMinValue = 0.0f;
    float mMaxValue = 6.0f;
};

} // namespace MNN

#endif

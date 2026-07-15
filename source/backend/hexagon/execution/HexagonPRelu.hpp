#ifndef HexagonPRelu_hpp
#define HexagonPRelu_hpp

#include "HexagonExecution.hpp"

namespace MNN {

class HexagonPRelu : public HexagonExecution {
public:
    static HexagonPRelu* create(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs);
    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    HexagonPRelu(Backend* backend, const Op* op, int pack);

    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    std::shared_ptr<Tensor> mSlope;
    int mSlopeCount = 0;
    int mPack = 4;
};

} // namespace MNN

#endif

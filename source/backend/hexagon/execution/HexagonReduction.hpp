#ifndef HexagonReduction_hpp
#define HexagonReduction_hpp

#include "HexagonExecution.hpp"

namespace MNN {

class HexagonReduction : public HexagonExecution {
public:
    static HexagonReduction* create(Backend* backend, const Op* op,
                                    const std::vector<Tensor*>& inputs,
                                    const std::vector<Tensor*>& outputs);
    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    HexagonReduction(Backend* backend, int opType, int axis, bool masked);
    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    int mOpType = 0;
    int mAxis = 1;
    bool mMasked = false;
};

} // namespace MNN

#endif

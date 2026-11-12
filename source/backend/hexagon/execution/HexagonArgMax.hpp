#ifndef HexagonArgMax_hpp
#define HexagonArgMax_hpp

#include "HexagonExecution.hpp"

namespace MNN {

// Rows of `rowSize` fp16 elements -> one int32 winning index per row, all on the DSP. Supports the
// DFlash draft/verify argmax, which would otherwise round-trip every row through the host each step.
class HexagonArgMax : public HexagonExecution {
public:
    virtual ~HexagonArgMax() = default;

    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    static HexagonArgMax* create(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs);

private:
    explicit HexagonArgMax(Backend* backend, int axis);

    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    int mAxis = -1;
};

} // namespace MNN

#endif

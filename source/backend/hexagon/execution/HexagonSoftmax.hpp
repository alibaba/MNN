#ifndef HexagonSoftmax_hpp
#define HexagonSoftmax_hpp

#include "HexagonExecution.hpp"

namespace MNN {

class HexagonSoftmax : public HexagonExecution {
public:
    virtual ~HexagonSoftmax() = default;

    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    static HexagonSoftmax* create(Backend* backend, const Op* op,
                                  const std::vector<Tensor*>& inputs,
                                  const std::vector<Tensor*>& outputs);

private:
    explicit HexagonSoftmax(Backend* backend, int axis);

    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    int mAxis = 0;
};

} // namespace MNN

#endif

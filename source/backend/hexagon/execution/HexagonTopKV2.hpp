#ifndef HexagonTopKV2_hpp
#define HexagonTopKV2_hpp

#include "HexagonExecution.hpp"

namespace MNN {

class HexagonTopKV2 : public HexagonExecution {
public:
    virtual ~HexagonTopKV2() = default;

    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    static HexagonTopKV2* create(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs);

private:
    explicit HexagonTopKV2(Backend* backend);

    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;
};

} // namespace MNN

#endif

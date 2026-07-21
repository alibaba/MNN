#ifndef HexagonSelect_hpp
#define HexagonSelect_hpp

#include <cstddef>

#include "HexagonExecution.hpp"

namespace MNN {

class HexagonSelect : public HexagonExecution {
public:
    virtual ~HexagonSelect() = default;

    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    static HexagonSelect* create(Backend* backend, const Op* op);

private:
    explicit HexagonSelect(Backend* backend);

    int mBytes = 2;
    int mCondBytes = 4;
    size_t mOutSize = 1;
    size_t mCondSize = 1;
    size_t mIn1Size = 1;
    size_t mIn2Size = 1;

    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;
};

} // namespace MNN

#endif

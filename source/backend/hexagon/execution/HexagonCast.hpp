#ifndef HexagonCast_hpp
#define HexagonCast_hpp

#include <cstdint>

#include "HexagonExecution.hpp"
#include "core/BufferAllocator.hpp"

namespace MNN {

class HexagonCast : public HexagonExecution {
public:
    virtual ~HexagonCast() = default;

    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    static HexagonCast* create(Backend* backend, const Op* op,
                               const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs);

private:
    explicit HexagonCast(Backend* backend, int castType);

    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    BufferAllocator* mAllocator = nullptr;
    int mCastType = 0;
};

} // namespace MNN

#endif

#ifndef HexagonUnary_hpp
#define HexagonUnary_hpp

#include <cstdint>

#include "core/BufferAllocator.hpp"
#include "HexagonExecution.hpp"

namespace MNN {

class HexagonUnary : public HexagonExecution {
public:
    virtual ~HexagonUnary() {
        // Do nothing
    }

    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    static HexagonUnary* create(Backend* backend, const Op* op);

private:
    explicit HexagonUnary(Backend* backend, int dspOpType);

    BufferAllocator* mAllocator = nullptr;


    int mBytes = 2;
    int mDspOpType = 0;
    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;
};

} // namespace MNN

#endif

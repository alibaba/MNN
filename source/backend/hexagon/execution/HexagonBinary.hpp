#ifndef HexagonBinary_hpp
#define HexagonBinary_hpp

#include <cstdint>

#include "core/BufferAllocator.hpp"
#include "HexagonExecution.hpp"

namespace MNN {

class HexagonBinary : public HexagonExecution {
public:
    struct BinaryRegion {
        int32_t src0Offset;
        int32_t src1Offset;
        int32_t dstOffset;
        int32_t size[3];
        int32_t src0Stride[3];
        int32_t src1Stride[3];
        int32_t dstStride[3];
    } __attribute__((packed));

    virtual ~HexagonBinary() {
        // Do nothing
    }

    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    static HexagonBinary* create(Backend* backend, const Op* op);

private:
    explicit HexagonBinary(Backend* backend, int dspOpType);

    BufferAllocator* mAllocator = nullptr;


    int mBytes = 2;
    int mDspOpType = 0;
    size_t mOutSize = 1;
    size_t mIn0Size = 1;
    size_t mIn1Size = 1;
    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;
};

} // namespace MNN

#endif

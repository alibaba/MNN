#ifndef HexagonTMac_hpp
#define HexagonTMac_hpp

#include "HexagonExecution.hpp"
#include "core/BufferAllocator.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class HexagonTMac : public HexagonExecution {
public:
    struct Resource {
        MemChunk weight;
        MemChunk scale;
        MemChunk bias;
        bool hasBias = false;
        int inputChannels = 0;
        int outputChannels = 0;
        int scaleBlockNum = 1;
        int asymmetric = 0;
        BufferAllocator* allocator = nullptr;
        ~Resource();
    };

    static HexagonTMac* create(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs);
    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    HexagonTMac(Backend* backend, std::shared_ptr<Resource> res, const Op* op);
    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

private:
    std::shared_ptr<Resource> mResource;
    int mRelu = 0;
    int mRelu6 = 0;
};

} // namespace MNN

#endif

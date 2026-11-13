#ifndef HexagonConvolutionInt8_hpp
#define HexagonConvolutionInt8_hpp

#include "HexagonExecution.hpp"
#include "core/BufferAllocator.hpp"

#include <memory>

namespace MNN {

class HexagonConvolutionInt8 : public HexagonExecution {
public:
    ~HexagonConvolutionInt8() override = default;

    bool onClone(Backend* backend, const Op* op, Execution** dst) override;

    static bool validTensorContract(const Op* op,
                                    const std::vector<Tensor*>& inputs,
                                    const std::vector<Tensor*>& outputs,
                                    int channelPack);
    static HexagonConvolutionInt8* create(
        Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
        const std::vector<Tensor*>& outputs);

private:
    struct Resource {
        BufferAllocator* allocator = nullptr;
        MemChunk weight;
        MemChunk bias;
        MemChunk requant;
        int weightBytes = 0;
        int biasBytes = 0;
        int requantBytes = 0;
        int inputChannels = 0;
        int outputChannels = 0;
        int channelPack = 0;
        int relu = 0;
        int relu6 = 0;
        ~Resource();
    };

    HexagonConvolutionInt8(Backend* backend,
                           std::shared_ptr<Resource> resource);

    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs,
                         const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    std::shared_ptr<Resource> mResource;
};

} // namespace MNN

#endif

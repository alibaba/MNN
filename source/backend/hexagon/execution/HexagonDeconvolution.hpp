#ifndef HexagonDeconvolution_hpp
#define HexagonDeconvolution_hpp

#include "HexagonExecution.hpp"
#include "core/BufferAllocator.hpp"
#include "core/ConvolutionCommon.hpp"

namespace MNN {

class HexagonDeconvolution : public HexagonExecution {
public:
    struct Resource {
        MemChunk weight;
        MemChunk bias;
        bool hasBias = false;
        int inputChannels = 0;
        int outputChannels = 0;
        int kernelX = 1;
        int kernelY = 1;
        BufferAllocator* allocator = nullptr;
        ~Resource();
    };

    static HexagonDeconvolution* create(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                                        const std::vector<Tensor*>& outputs);
    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    HexagonDeconvolution(Backend* backend, std::shared_ptr<Resource> res, const Op* op);
    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;

    std::shared_ptr<Resource> mResource;
    ConvolutionCommon::Im2ColParameter mParam;
    const Op* mOp = nullptr;
};

} // namespace MNN

#endif

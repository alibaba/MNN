#ifndef HexagonConvolution_hpp
#define HexagonConvolution_hpp
#include "core/BufferAllocator.hpp"
#include "core/ConvolutionCommon.hpp"
#include "HexagonExecution.hpp"
namespace MNN {
class HexagonConvolution : public HexagonExecution {
public:
    struct Resource {
        MemChunk weight;
        MemChunk bias;
        bool hasBias = false;
        // Optional INT4 (W4A16) packed weight buffer (my_block_q4_0 sequence)
        MemChunk int4Weight;
        // Row-major INT4 + fp16 scale copy for GatherV2 clones.
        MemChunk gatherInt4Weight;
        // Flag and metadata for INT4 path
        bool useInt4W4A16 = false;
        int int4WeightType = 0;  // ggml_type, e.g. GGML_TYPE_Q4_0
        int int4LayoutType = 0;  // 1: per-tile permuted
        int int4ScaleBlockNum = 1;
        int gatherInputChannels = 0;
        int gatherOutputChannels = 0;
        BufferAllocator* allocator;
        ~ Resource();
    };
    virtual ~HexagonConvolution() = default;

    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    static HexagonConvolution* create(Backend *backend, const Op* op);
private:
    ErrorCode onBuildCmd(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                         std::vector<HexagonCommand>& dst) override;
    HexagonConvolution(Backend *backend, std::shared_ptr<Resource> res, const Op* op);
    std::shared_ptr<Resource> mResource;
    ConvolutionCommon::Im2ColParameter mParam;
    int mMp = 1;
    int mNp = 1;
    int mKp = 1;
    bool mUseIm2Col = false;
    int mKernelY = 1;
    int mKernelX = 1;
    int mStrideY = 1;
    int mStrideX = 1;
    int mDilateY = 1;
    int mDilateX = 1;
    int mRelu = 0;
    int mRelu6 = 0;
    const Op* mOp = nullptr;
    std::shared_ptr<Tensor> mTempInTensor;
    std::shared_ptr<Tensor> mTempOutTensor;
};
};

#endif

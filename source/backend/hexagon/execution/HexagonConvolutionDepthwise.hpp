#ifndef HexagonConvolutionDepthwise_hpp
#define HexagonConvolutionDepthwise_hpp

#include "core/BufferAllocator.hpp"
#include "core/ConvolutionCommon.hpp"
#include "HexagonExecution.hpp"

namespace MNN {

class HexagonConvolutionDepthwise : public HexagonExecution {
public:
    struct Resource {
        MemChunk weight;
        MemChunk bias;
        BufferAllocator* allocator;
        ~Resource();
    };

    virtual ~HexagonConvolutionDepthwise() = default;

    bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    static HexagonConvolutionDepthwise* create(Backend* backend, const Op* op);

private:
    HexagonConvolutionDepthwise(Backend* backend, std::shared_ptr<Resource> res, const Convolution2DCommon* common);
    HexagonConvolutionDepthwise(Backend* backend, std::shared_ptr<Resource> res);

    std::shared_ptr<Resource> mResource;
    const Convolution2DCommon* mCommon = nullptr;

    int mKernelX = 0;
    int mKernelY = 0;
    int mStrideX = 0;
    int mStrideY = 0;
    int mDilateX = 0;
    int mDilateY = 0;
    int mPadX = 0;
    int mPadY = 0;
    int mPadMode = 0;

    int mRelu = 0;
    int mRelu6 = 0;

    int mBatch = 0;
    int mInputHeight = 0;
    int mInputWidth = 0;
    int mOutputHeight = 0;
    int mOutputWidth = 0;

    int mChannel = 0;
    int mChannelBlock = 0;
    int mPack = 4;
    int mBytes = 2;
    ErrorCode onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                         std::vector<HexagonCommand>& dst) override;
};

} // namespace MNN

#endif

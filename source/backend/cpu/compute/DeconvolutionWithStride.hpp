//
//  DeconvolutionWithStride.hpp
//  MNN
//
//  Created by MNN on 2018/10/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DeconvolutionWithStride_hpp
#define DeconvolutionWithStride_hpp

#include "backend/cpu/CPUDeconvolution.hpp"
#include "core/Backend.hpp"
#include <mutex>
namespace MNN {
class DeconvolutionWithStride : public CPUDeconvolutionCommon {
public:
    DeconvolutionWithStride(const Tensor *input, const Op *convOp, Backend *b);
    virtual ~DeconvolutionWithStride();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    struct ComputeUnit {
        std::shared_ptr<Tensor> weight;
        std::shared_ptr<Tensor> dstBuffer;
        int xUnit   = 0;
        int yUnit   = 0;
        int xOffset = 0;
        int yOffset = 0;

        struct Winograd {
            std::shared_ptr<Tensor> dstTransformedBuffer;

            std::shared_ptr<Tensor> A;
            std::shared_ptr<Tensor> B;
            std::shared_ptr<Tensor> G;

            int srcUnitX = 0;
            int srcUnitY = 0;

            bool open = false;
        };

        Winograd winogradInfo;
    };

private:
    bool _alloc(Backend::StorageType type);
    void _release(Backend::StorageType type);
    void _extract(const Op *convOp);

    std::shared_ptr<Tensor> mSrcBuffer;
    std::shared_ptr<Tensor> mMatMulPackBuffer;
    std::map<int, std::shared_ptr<Tensor>> mTransformedBuffer;
    std::shared_ptr<Tensor> mDestBuffer;

    std::vector<ComputeUnit> mComputeUnits;

    std::mutex mLock;
    int mStrideX = 1;
    int mStrideY = 1;
    std::vector<float> mPostParameters;
};
} // namespace MNN

#endif /* DeconvolutionWithStride_hpp */

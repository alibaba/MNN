//
//  ConvSubgroupBufExecution.hpp
//  MNN
//
//  Created by MNN on 2023/07/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP

#ifndef ConvSubgroupBufExecution_hpp
#define ConvSubgroupBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"
namespace MNN {
namespace OpenCL {
struct ConvSubgroupBufResource {
    const Convolution2DCommon* mConv2dCommonParams;
    const Convolution2D* mConv2dParams;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mDilations{1, 1};
    std::shared_ptr<Tensor> mFilter;
    std::shared_ptr<Tensor> mBias;
    int mKernelWidth;
    int mKernelHeight;
    int mOutputChannel;
    int mInputChannel;
    std::set<std::string> mBuildOptions;
};

class ConvSubgroupBuf : public CommonExecution {
public:
    virtual ~ConvSubgroupBuf();

    ConvSubgroupBuf(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                    Backend* backend);
    ConvSubgroupBuf(std::shared_ptr<ConvSubgroupBufResource> resource, const MNN::Op* op, Backend* backend);

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    static bool valid(const Convolution2DCommon* common, const Tensor* input, const Tensor* output, int maxWidth,
                      int maxHeight, int limit = 8192);

private:
    void transformWeight(const Tensor* weightDest, const Tensor* source);

    OpenCLBackend* mOpenCLBackend;
    std::shared_ptr<ConvSubgroupBufResource> mResource;
    std::vector<int> mPaddings{0, 0};
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    std::vector<uint32_t> mTranseGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mTranseLocalWorkSize{1, 1, 1, 1};
    std::shared_ptr<Tensor> mSource;
    bool mNeedTranse = false;
    
};

} // namespace OpenCL
} // namespace MNN
#endif /* ConvSubgroupBufExecution_hpp */
#endif /* MNN_SUPPORT_INTEL_SUBGROUP */
#endif /* MNN_OPENCL_BUFFER_CLOSED */

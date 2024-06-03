//
//  ConvBufExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef ConvBufExecution_hpp
#define ConvBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"
namespace MNN {
namespace OpenCL {

struct ConvBufResource {
    const Convolution2DCommon *mConv2dCommonParams;
    const Convolution2D *mConv2dParams;
    std::shared_ptr<cl::Buffer> mKernelBuffer;
    std::shared_ptr<cl::Image2D> mKernelImage;
    std::shared_ptr<Tensor> dequantScale;
    std::shared_ptr<Tensor> dequantOffset;
    std::shared_ptr<Tensor> mFilter;
    std::shared_ptr<Tensor> mBias;
    int mKernelWidth;
    int mKernelHeight;
    int mOutputChannel;
    int mInputChannel;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mDilations{1, 1};
    std::set<std::string> mBuildOptions;
    bool mConv1x1Opt = false;
    bool mConv1x1C8Opt = false;
    /*
     0 -> not use
     1 -> use small tile
     2 -> use quieter large tile
     */
    int mConvGemmOptLevel = 0;
    std::shared_ptr<Execution> mRasterExe;
    bool mUseImage = false;
};

class ConvBufCommonExecution {
public:
    ConvBufCommonExecution(Backend *backend);
    ConvBufCommonExecution(const Convolution2D *op, Backend *backend);
    virtual ~ConvBufCommonExecution();

    std::pair<std::vector<uint32_t>,  uint32_t> gws2dLwsTune(const std::shared_ptr<KernelWrap> &kernel, const std::vector<uint32_t> &gws, const std::string &kernelName, const uint32_t maxWorkGroupSize);
protected:
    std::shared_ptr<ConvBufResource> mResource;
    OpenCLBackend *mOpenCLBackend;
};

class ConvBufExecution : public ConvBufCommonExecution, public CommonExecution {
public:
    ConvBufExecution(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const MNN::Op *op, Backend *backend);
    ConvBufExecution(std::shared_ptr<ConvBufResource> resource, const MNN::Op* op, Backend* backend);
    virtual ~ConvBufExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    void setConv1x1WeightBuffer(int packCout, int packCin, const float* filterDataPtr);
private:
    void _generateFilterConvertRegion(Tensor *virtualFilter, Tensor *originBuffer) const;

    std::vector<int> mPaddings{0, 0};
    std::vector<uint32_t> mGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mLocalWorkSize{1, 1, 1, 1};
    std::shared_ptr<KernelWrap> mKernel;
    std::shared_ptr<Tensor> mConvGemmInpTensor;
    std::shared_ptr<Tensor> mConvGemmOutTensor;
    std::shared_ptr<KernelWrap> mPreKernel = nullptr;
    std::vector<uint32_t> mPreGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mPreLocalWorkSize{1, 1, 1, 1};
    std::shared_ptr<KernelWrap> mPostKernel = nullptr;
    std::vector<uint32_t> mPostGlobalWorkSize{1, 1, 1};
    std::vector<uint32_t> mPostLocalWorkSize{1, 1, 1, 1};
    const float* mFilterDataPtr = nullptr;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ConvBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */

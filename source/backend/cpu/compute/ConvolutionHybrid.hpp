//
//  ConvolutionHybrid.hpp
//  MNN
//
//  Created by MNN on 2023/10/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ConvolutionHybrid_hpp
#define ConvolutionHybrid_hpp

#include <functional>
#include "backend/cpu/CPUConvolution.hpp"

typedef void(*LowMemoryGemmFuncWithInt8Weight)(float* C, const int8_t* A, const int8_t* B, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, size_t realSize, const float** param);
namespace MNN {
class ConvolutionHybrid : public CPUConvolution {
public:
    ConvolutionHybrid(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                      size_t originWeightSize, const float *bias, size_t biasSize, std::shared_ptr<ConvolutionCommon::Int8Common>);
    ConvolutionHybrid(std::shared_ptr<CPUConvolution::Resource> resource, const Convolution2DCommon *common, Backend* b);
    static bool initQuantizeResource(std::shared_ptr<ConvolutionCommon::Int8Common> int8Info, std::shared_ptr<CPUConvolution::Resource> resource, int hU, int hP, int lU, int lP, int outputCount, int srcChannel, int kernelSize, int bytes);

    virtual ~ConvolutionHybrid();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    ErrorCode allocTensor(Tensor* tensor, size_t size);
    ErrorCode allocDynamicQuantInfo(int thread, int batch, int ic, int oc, int bytes);
private:
    struct DynamicQuantInfo {
        Tensor quant_info;
        Tensor quant_buffer;
    };
    std::shared_ptr<CPUConvolution::Resource> mResource;
    std::function<void()> mDynamicQuant;
    std::pair<int, std::function<void(int)>> mFunction;
    DynamicQuantInfo mQuantInfo;
    bool ANeedToPack8 = false;
    std::shared_ptr<Tensor> mInputTemp;
    std::shared_ptr<Tensor> mOutputTemp;
};
} // namespace MNN

#endif /* ConvolutionHybrid_hpp */

//
//  MetalSharedGather.hpp
//  MNN

#ifndef MetalSharedGather_hpp
#define MetalSharedGather_hpp

#import "MetalExecution.hpp"
#import "MNN_generated.h"

#if MNN_METAL_ENABLED
namespace MNN {

// SharedGather implementation on Metal backend.
// It reuses quantized 1x1 convolution weights and dequantization parameters
// to gather selected output-channel rows into a dense floating-point matrix.
class MetalSharedGather : public MetalExecution {
public:
    MetalSharedGather(Backend *backend,
                      int oc,
                      std::shared_ptr<Tensor> weight,
                      std::shared_ptr<Tensor> dequantScaleBias,
                      int dequantBits,
                      float scaleCoef);
    virtual ~MetalSharedGather() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) override;

    virtual void onEncode(const std::vector<Tensor *> &inputs,
                          const std::vector<Tensor *> &outputs,
                          id<MTLComputeCommandEncoder> encoder) override;

    virtual bool onClone(Backend *bn, const Op *op, Execution **dst) override;

private:
    // conv1x1_constants host-side mirror, used by conv1x1_w_dequant and shared_gather kernels
    struct Conv1x1Constants {
        int input_size;      // repurposed as ic for SharedGather
        int input_slice;     // ic_4
        int output_width;    // selectSize
        int output_height;   // unused for SharedGather
        int output_size;     // total elements (selectSize * ic)
        int output_slice;    // oc_4
        int output_channel;  // oc
        int batch;           // unused for SharedGather
        int block_size;      // quant block size along K axis
        int activation;      // not used (no activation)
        float scale_coef;    // scale normalization factor
    };

private:
    int mOc = 0;                    // number of output channels (rows in weight)
    std::shared_ptr<Tensor> mWeight;            // quantized int8/4 weight
    std::shared_ptr<Tensor> mDequantScaleBias;  // packed scale + bias
    int mDequantBits = 0;           // 4 or 8
    float mScaleCoef = 1.0f;

    // direct int4/int8 quant gather pipeline (preferred path)
    id<MTLComputePipelineState> mQuantPipeline = nil;
    std::pair<MTLSize, MTLSize> mQuantThreads;

    // constant buffer shared by dequant and gather kernels
    id<MTLBuffer> mConstBuffer = nil;
};

} // namespace MNN
#endif /* MNN_METAL_ENABLED */

#endif /* MetalSharedGather_hpp */

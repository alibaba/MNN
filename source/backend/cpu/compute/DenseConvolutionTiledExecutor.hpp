//
//  DenseConvolutionTiledExecutor
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef DenseConvolutionTiledExecutor_hpp
#define DenseConvolutionTiledExecutor_hpp


#include <functional>
#include "backend/cpu/CPUConvolution.hpp"
#include "ConvolutionTiledExecutor.hpp"
// Tiled Slide Window or Im2Col + GEMM
namespace MNN {
typedef void(*lowMemoryMatmulUnit)(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
typedef void(*lowMemoryMatmulRemain)(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
class DenseConvolutionTiledImpl : public ConvolutionTiledImpl {
public:
    DenseConvolutionTiledImpl(const Convolution2DCommon *common, Backend *b, CPUConvolution::Resource* resource = nullptr) : ConvolutionTiledImpl(common, b) {
        mResource = resource;
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs,
                                         const std::vector<Tensor*>& outputs) override;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs,
                                         const std::vector<Tensor*>& outputs) override;
    virtual ~DenseConvolutionTiledImpl() = default;
    void getPackParameter(int* eP, int* lP, int* hP, const CoreFunctions* core) override;
    static PerfConfig bestTileConvolutionConfig(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b);
protected:
};
class DenseConvolutionTiledExecutor : public ConvolutionTiledExecutor {
public:
    DenseConvolutionTiledExecutor(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                                  size_t originWeightSize, const float *bias, size_t biasSize, std::shared_ptr<ConvolutionCommon::Int8Common>);

    DenseConvolutionTiledExecutor(std::shared_ptr<CPUConvolution::Resource> res, const Convolution2DCommon *common, Backend* b);
    virtual ~DenseConvolutionTiledExecutor();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    void initWeight(float *dest, const float *source, float* cache, int depth, int outputCount, int kernelSize, const CoreFunctions* function);
    static PerfConfig bestTileConvolutionConfig(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b) {
        return DenseConvolutionTiledImpl::bestTileConvolutionConfig(common, inputTensor, outputTensor, threadNumber, b);
    }
    static bool initQuantizeResource(std::shared_ptr<ConvolutionCommon::Int8Common> int8Info, std::shared_ptr<CPUConvolution::Resource> resource, int hU, int hP, int lU, int lP, int outputCount, int srcChannel, int kernelSize, int bytes);
    static void selectLowMemoryMatmulFunc(lowMemoryMatmulUnit* matmulUnit, lowMemoryMatmulRemain* matmulRemain, float* weightBytes, int32_t weightQuantBits, const CoreFunctions* core);
    struct DequantizeCache {
        std::shared_ptr<MNN::Tensor> weight;
        std::shared_ptr<MNN::Tensor> weightInt8;
    };
protected:
    DequantizeCache mWeightCache;
    std::shared_ptr<DenseConvolutionTiledImpl> mProxy;
};

class ConvolutionTiledExecutorMultiInput : public Execution {
public:
    ConvolutionTiledExecutorMultiInput(const Convolution2DCommon *common, Backend *b) : Execution(b) {
        mProxy.reset(new DenseConvolutionTiledImpl(common, b));
    }
    virtual ~ConvolutionTiledExecutorMultiInput() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mTempWeight;
    std::shared_ptr<Tensor> mTempWeightCache;
    std::shared_ptr<Tensor> mTempBias;
    std::shared_ptr<DenseConvolutionTiledImpl> mProxy;
    std::vector<Tensor *> mInputs;
};
} // namespace MNN

#endif /* DenseConvolutionTiledExecutor_hpp */

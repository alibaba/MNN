//
//  LayerNormBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/07/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/buffer/LayerNormBufExecution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

LayerNormBufExecution::LayerNormBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto runtime   = mOpenCLBackend->getOpenCLRuntime();
    const auto* layer_norm_param = op->main_as_LayerNorm();
    axis_size = layer_norm_param->axis()->size();
    epsilon_ = layer_norm_param->epsilon();
    group_ = layer_norm_param->group();
    auto bufferUnitSize = runtime->isSupportedFP16() ? sizeof(half_float::half) : sizeof(float);

    if(layer_norm_param->gamma() && layer_norm_param->beta()){
        has_gamma_beta_ = true;
        {
            auto error = CL_SUCCESS;
            int size = layer_norm_param->gamma()->size();
            mGammaBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, ALIGN_UP4(size) * bufferUnitSize));
            auto GammaPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mGammaBuffer.get()), true, CL_MAP_WRITE, 0, ALIGN_UP4(size) * bufferUnitSize, nullptr, nullptr, &error);
            const float* gamma_data = layer_norm_param->gamma()->data();
            if(GammaPtrCL != nullptr && error == CL_SUCCESS){
                if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
                    for (int i = 0; i < size; i++)
                    {
                        ((half_float::half*)GammaPtrCL)[i] = (half_float::half)(gamma_data[i]);
                    }
                    for(int i=size; i<ALIGN_UP4(size); i++) {
                        ((half_float::half*)GammaPtrCL)[i] = (half_float::half)(0.0f);
                    }
                }else{
                    ::memset(GammaPtrCL, 0, ALIGN_UP4(size) * sizeof(float));
                    ::memcpy(GammaPtrCL, gamma_data, size * sizeof(float));
                }
            }else{
                MNN_ERROR("Map error GammaPtrCL == nullptr \n");
            }
            mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mGammaBuffer.get(), GammaPtrCL);
        }
        {
            auto error = CL_SUCCESS;
            int size = layer_norm_param->beta()->size();
            mBetaBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, ALIGN_UP4(size) * bufferUnitSize));
            auto BetaPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mBetaBuffer.get()), true, CL_MAP_WRITE, 0, ALIGN_UP4(size) * bufferUnitSize, nullptr, nullptr, &error);
            const float* beta_data = layer_norm_param->beta()->data();
            if(BetaPtrCL != nullptr && error == CL_SUCCESS){
                if(mOpenCLBackend->getOpenCLRuntime()->isSupportedFP16()){
                    for (int i = 0; i < size; i++)
                    {
                        ((half_float::half*)BetaPtrCL)[i] = (half_float::half)(beta_data[i]);
                    }
                    for(int i=size; i<ALIGN_UP4(size); i++) {
                        ((half_float::half*)BetaPtrCL)[i] = (half_float::half)(0.0f);
                    }
                }else{
                    ::memset(BetaPtrCL, 0, ALIGN_UP4(size) * sizeof(float));
                    ::memcpy(BetaPtrCL, beta_data, size * sizeof(float));
                }
            }else{
                MNN_ERROR("Map error BetaPtrCL == nullptr \n");
            }
            mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mBetaBuffer.get(), BetaPtrCL);
        }
    }
}

int LayerNormBufExecution::getLocalSize(int size, int maxGroupSize){
    int local_size = 1;
    while(local_size * 2 <= maxGroupSize && local_size * 2 <= size){
        local_size *= 2;
    }
    return local_size;
}

ErrorCode LayerNormBufExecution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];
    auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    const int inputBatch    = inputShape[0];
    const int inputHeight   = inputShape[1];
    const int inputWidth    = inputShape[2];
    const int inputChannels = inputShape[3];
    auto MaxWorkItems = runtime->getMaxWorkItemSizes();
    int local_size;
    int rank = inputs.at(0)->dimensions();
    int outter_size = 1;
    int inner_size = 1;
    for (int i = 0; i < rank - axis_size; ++i) {
        outter_size *= inputs.at(0)->length(i);
    }
    for (int i = rank - axis_size; i < rank; ++i) {
        inner_size *= inputs.at(0)->length(i);
    }

    
    std::set<std::string> buildOptions;
    if(has_gamma_beta_){
        buildOptions.emplace("-DGAMMA_BETA");
    }
    std::string kernelName;
    if (inner_size == inputWidth && outter_size == inputBatch * inputHeight * inputChannels) {
        kernelName = "layernorm_w_buf";
        local_size = getLocalSize(inputWidth, MaxWorkItems[0]);
        buildOptions.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
        mKernel = runtime->buildKernel("layernorm_buf", kernelName, buildOptions);
        
        mGWS = {static_cast<uint32_t>(local_size),
                static_cast<uint32_t>(inputHeight * UP_DIV(inputChannels, 4)),
                static_cast<uint32_t>(inputBatch)};
    }else if(inner_size == inputWidth * inputHeight && outter_size == inputBatch * inputChannels){
        kernelName = "layernorm_hw_buf";
        local_size = getLocalSize(inputWidth * inputHeight, MaxWorkItems[0]);
        buildOptions.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
        mKernel = runtime->buildKernel("layernorm_buf", kernelName, buildOptions);
        
        mGWS = {static_cast<uint32_t>(local_size),
                static_cast<uint32_t>(UP_DIV(inputChannels, 4)),
                static_cast<uint32_t>(inputBatch)};
    }else if(inner_size == inputWidth * inputHeight * inputChannels && outter_size == inputBatch){
        kernelName = "layernorm_chw_buf";
        local_size = getLocalSize(inputWidth * inputHeight, MaxWorkItems[0]);
        buildOptions.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
        mKernel = runtime->buildKernel("layernorm_buf", kernelName, buildOptions);
        
        mGWS = {static_cast<uint32_t>(local_size),
                static_cast<uint32_t>(1),
                static_cast<uint32_t>(inputBatch)};
    }
    mLWS = {static_cast<uint32_t>(local_size), 1, 1};

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= mKernel.setArg(idx++, mGWS[0]);
    ret |= mKernel.setArg(idx++, mGWS[1]);
    ret |= mKernel.setArg(idx++, mGWS[2]);
    ret |= mKernel.setArg(idx++, openCLBuffer(input));
    ret |= mKernel.setArg(idx++, openCLBuffer(output));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(inputWidth));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(inputHeight));
    ret |= mKernel.setArg(idx++, static_cast<int32_t>(inputChannels));
    if(has_gamma_beta_){
        ret |= mKernel.setArg(idx++, *mGammaBuffer.get());
        ret |= mKernel.setArg(idx++, *mBetaBuffer.get());
    }
    ret |= mKernel.setArg(idx++, epsilon_);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LayerNormBufExecution");

    return NO_ERROR;

}

ErrorCode LayerNormBufExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start LayerNormBufExecution onExecute... \n");
#endif

#ifdef ENABLE_OPENCL_TIME_PROFILER
    cl::Event event;
    run3DKernelDefault(mKernel, mGWS, mLWS,
                       mOpenCLBackend->getOpenCLRuntime(), &event);
    
    mOpenCLBackend->getOpenCLRuntime()->pushEvent({"LayerNormBuf", event});
#else
    run3DKernelDefault(mKernel, mGWS, mLWS, mOpenCLBackend->getOpenCLRuntime());
#endif

#ifdef LOG_VERBOSE
    MNN_PRINT("end LayerNormBufExecution onExecute... \n");
#endif

    return NO_ERROR;
}

class LayerNormBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~LayerNormBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
		for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        const auto* layer_norm_param = op->main_as_LayerNorm();
        int axis_size = layer_norm_param->axis()->size();
        int group = layer_norm_param->group();
        if(group > 1){
			return nullptr;
        }
    	return new LayerNormBufExecution(inputs, op, backend);
    }
};

OpenCLCreatorRegister<LayerNormBufCreator> __LayerNormBuf_op_(OpType_LayerNorm, BUFFER);

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_OPENCL_BUFFER_CLOSED */

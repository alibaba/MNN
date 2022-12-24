//
//  DeconvSingleInputExecution.cpp
//  MNN
//
//  Created by MNN on 2022/03/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "DeconvSingleInputExecution.hpp"

namespace MNN {
namespace CUDA {

template<typename T>
__global__ void DeconvKernelReorder(const float* B, T* BP, int kw, int kh, int ic, int oc, int icPack) {
    int kernelCount = kw * kh;
    int e = oc * kernelCount;
    int l = ic;
    int lAlign = icPack;

    int maxCount = e * lAlign;
    // l * e  --> e * lp
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        int lp_idx = index % lAlign;
        int e_idx = index / lAlign;

        if(lp_idx >= l) {
            BP[index] = (T)0.0f;
            continue;
        }
        BP[index] = (T)(B[lp_idx * e + e_idx]);
    }
}


__global__ void __Float22Half2(const float* param,
    half* output,
    const size_t maxCount
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += blockDim.x * gridDim.x) {
        float2* srcPtr = (float2 *)(param + (index << 2));
        half2* dstPtr = (half2*)(output + (index << 2));
        dstPtr[0] = __float22half2_rn(srcPtr[0]);
        dstPtr[1] = __float22half2_rn(srcPtr[1]);
    }
}

template <typename Dtype>
__global__ void Col2Im(const int n, const Dtype* data_col,
    const int batch, const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    const float* bias, Dtype* data_im
) {
    const int channel_pack = ((channels+7) / 8) * 8;

    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x) {
        Dtype val = 0;        
        const int c_p = index % channel_pack;
        const int idx_tmp = index / channel_pack;
        const int b_im = idx_tmp / (width * height);
        const int hw = idx_tmp % (width * height);
        const int c_im = c_p;
        const int w_im = hw % width + pad_w;
        const int h_im = hw / width + pad_h;

        if(c_im >= channels) {
            data_im[index] = val;
            break;
        }
        if(nullptr != bias) {
            val += (Dtype)bias[c_im];
        }
        int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
        int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
        // compute the start and end of the output
        const int w_col_start =
            (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
        const int w_col_end = min(w_im / stride_w + 1, width_col);
        const int h_col_start =
            (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
        const int h_col_end = min(h_im / stride_h + 1, height_col);
        // TODO: use LCM of stride and dilation to avoid unnecessary loops
        for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
            for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                int h_k = (h_im - h_col * stride_h);
                int w_k = (w_im - w_col * stride_w);
                if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                    h_k /= dilation_h;
                    w_k /= dilation_w;

                    const int data_col_index = ((((c_im * kernel_h + h_k) * kernel_w + w_k) * batch + b_im) *
                                            height_col + h_col) * width_col + w_col;
                    val += data_col[data_col_index];
                }
            }
        }
        data_im[index] = val;
    }
}


DeconvSingleInputExecution::Resource::Resource(Backend* bn, const MNN::Op* op) {
    mBackend = bn;
    auto runtime = static_cast<CUDABackend*>(bn)->getCUDARuntime();
    
    auto conv       = op->main_as_Convolution2D();
    auto common     = conv->common();
    mKernelInfo.kernelX        = common->kernelX();
    mKernelInfo.kernelY        = common->kernelY();
    mKernelInfo.groups         = common->group();
    mKernelInfo.strideX        = common->strideX();
    mKernelInfo.strideY        = common->strideY();
    mKernelInfo.dilateX        = common->dilateX();
    mKernelInfo.dilateY        = common->dilateY();
    mKernelInfo.activationType = common->relu() ? 1 : (common->relu6() ? 2 : 0);

    //weight host->device
    const float* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, conv, &filterDataPtr, &weightSize);
    mKernelInfo.kernelN = common->outputCount();
    mKernelInfo.kernelC = weightSize / mKernelInfo.kernelN / mKernelInfo.kernelX / mKernelInfo.kernelY;

    CutlassGemmInfo param;
    int e = mKernelInfo.kernelN * mKernelInfo.kernelX * mKernelInfo.kernelY;
    int l = mKernelInfo.kernelC;
    param.elh[0] = e;
    param.elh[1] = l;
    param.elhPad[0] = UP_DIV(e, PACK_NUMBER) * PACK_NUMBER;
    param.elhPad[1] = UP_DIV(l, PACK_NUMBER) * PACK_NUMBER;

    auto tempCacheBuffer = static_cast<CUDABackend*>(bn)->getStaticBufferPool()->alloc(weightSize * sizeof(float));
    float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
    runtime->memcpy(cacheWeight, filterDataPtr, weightSize * sizeof(float), MNNMemcpyHostToDevice);
    
    // Reorder weight
    if(static_cast<CUDABackend*>(bn)->getPrecision() == 1) {
        weightTensor.reset(Tensor::createDevice<int32_t>({param.elh[0] * param.elhPad[1]}));
    } else {
        weightTensor.reset(Tensor::createDevice<int16_t>({param.elh[0] * param.elhPad[1]}));
    }
    bn->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
    mFilter = (void *)weightTensor.get()->buffer().device;    
    
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    int threadNumbers = prop.maxThreadsPerBlock;
    if(static_cast<CUDABackend*>(bn)->getPrecision() == 1) {
        DeconvKernelReorder<<<cores, threadNumbers>>>((float*)cacheWeight, (float*)mFilter,
            mKernelInfo.kernelX, mKernelInfo.kernelY, mKernelInfo.kernelC, mKernelInfo.kernelN, param.elhPad[1]);
        checkKernelErrors;
    } else {
        DeconvKernelReorder<<<cores, threadNumbers>>>((float*)cacheWeight, (half*)mFilter,
            mKernelInfo.kernelX, mKernelInfo.kernelY, mKernelInfo.kernelC, mKernelInfo.kernelN, param.elhPad[1]);
        checkKernelErrors;
    }
    static_cast<CUDABackend*>(bn)->getStaticBufferPool()->free(tempCacheBuffer);

    // Copy Bias
    int biasSize = conv->bias()->size();
    biasTensor.reset(Tensor::createDevice<float>({biasSize}));
    bn->onAcquireBuffer(biasTensor.get(), Backend::STATIC);
    mBias = (void *)biasTensor.get()->buffer().device;
    cuda_check(cudaMemcpy(mBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));
    
}

DeconvSingleInputExecution::Resource::~Resource() {
    // Do nothing
}
DeconvSingleInputExecution::DeconvSingleInputExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res) : Execution(backend), mOp(op) {
    mResource = res;
    int precisonLevel = static_cast<CUDABackend*>(backend)->getPrecision();
    mFp16Infer = (precisonLevel == 2);
    mFp32Infer = (precisonLevel == 1);
    mFp16Fp32MixInfer = (precisonLevel == 0);
}

DeconvSingleInputExecution::~DeconvSingleInputExecution() {
    // Do nothing
}
bool DeconvSingleInputExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new DeconvSingleInputExecution(bn, op, mResource);
    *dst = dstExe;
    return true;
}


ErrorCode DeconvSingleInputExecution::onResize(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto input = inputs[0], output = outputs[0];
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);
    auto convCommon = mOp->main_as_Convolution2D()->common();

    // Col2Im Param
    auto pad = ConvolutionCommon::convolutionTransposePad(input, output, mOp->main_as_Convolution2D()->common());
    mCol2ImParamter.dilateX         = convCommon->dilateX();
    mCol2ImParamter.dilateY         = convCommon->dilateY();
    mCol2ImParamter.strideX         = convCommon->strideX();
    mCol2ImParamter.strideY         = convCommon->strideY();
    mCol2ImParamter.ic              = input->channel();
    mCol2ImParamter.oc              = output->channel();
    mCol2ImParamter.kernelX         = convCommon->kernelX();
    mCol2ImParamter.kernelY         = convCommon->kernelY();
    mCol2ImParamter.padX = pad.first;
    mCol2ImParamter.padY = pad.second;

    mCol2ImParamter.ih = input->height();
    mCol2ImParamter.iw = input->width();
    mCol2ImParamter.oh = output->height();
    mCol2ImParamter.ow = output->width();
    mCol2ImParamter.ob = output->batch();

    mActivationType = convCommon->relu() ? 1 : convCommon->relu6() ? 2 : 0;

    // Matmul Param
    int e = output->channel() * mCol2ImParamter.kernelX * mCol2ImParamter.kernelY;
    int l = input->channel();
    int h = input->height() * input->width() * output->batch();

    mGemmInfo.elh[0] = e;
    mGemmInfo.elh[1] = l;
    mGemmInfo.elh[2] = h;
    mGemmInfo.elhPad[0] = UP_DIV(e, PACK_NUMBER) * PACK_NUMBER;
    mGemmInfo.elhPad[1] = UP_DIV(l, PACK_NUMBER) * PACK_NUMBER;
    mGemmInfo.elhPad[2] = UP_DIV(h, PACK_NUMBER) * PACK_NUMBER;


    // Alloc temp cuda memory
    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    std::pair<void*, size_t> buffer_input, buffer_im2col;
    if(mFp16Fp32MixInfer) {
        buffer_input = pool->alloc(sizeof(__half) * mGemmInfo.elhPad[1] * mGemmInfo.elh[2]);
        mInputBuffer = (void*)((uint8_t*)buffer_input.first + buffer_input.second);
    } else {
        mInputBuffer = (void*)input->deviceId();
    }
    buffer_im2col = pool->alloc(bytes * mGemmInfo.elh[0] * mGemmInfo.elhPad[2]);
    mIm2ColBuffer = (__half*)((uint8_t*)buffer_im2col.first + buffer_im2col.second);
    if(mFp16Fp32MixInfer) {
        pool->free(buffer_input);
    }
    pool->free(buffer_im2col);

    if(mFp16Fp32MixInfer || mFp32Infer) {
        mZeroTensor.reset(Tensor::createDevice<uint32_t>({mGemmInfo.elhPad[2]}));
    } else {
        mZeroTensor.reset(Tensor::createDevice<uint16_t>({mGemmInfo.elhPad[2]}));
    }
    static_cast<CUDABackend*>(backend())->onAcquireBuffer(mZeroTensor.get(), Backend::STATIC);

    mZeroPtr = (void *)mZeroTensor.get()->buffer().device;
    cuda_check(cudaMemset(mZeroPtr, 0, mGemmInfo.elhPad[2]*bytes));

    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;
    cutlass::gemm::GemmCoord problem_size(mGemmInfo.elh[0], mGemmInfo.elh[2], mGemmInfo.elhPad[1]);// m n k

    if(mFp32Infer) {
        if(mActivationType == 1) {
            // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmCuda_F32_F32_Relu_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {(ElementInput_F32 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            {(ElementInput_F32 *)mInputBuffer, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            {(ElementOutput_F32 *)mZeroPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                            {(ElementOutput_F32 *)mIm2ColBuffer, mGemmInfo.elh[2]},  //  Ptr + ldm
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmCuda_F32_F32_Relu_AlignCuda::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmCudaF32F32Relu.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmCudaF32F32Relu.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
    
        } else if(mActivationType == 2) {
            // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmCuda_F32_F32_Relu6_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {(ElementInput_F32 *)mResource->mFilter, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F32 *)mInputBuffer, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mZeroPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)mIm2ColBuffer, mGemmInfo.elh[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmCuda_F32_F32_Relu6_AlignCuda::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmCudaF32F32Relu6.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmCudaF32F32Relu6.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
    
        } else {

            typename GemmCuda_F32_F32_Linear_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {(ElementInput_F32 *)mResource->mFilter, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F32 *)mInputBuffer, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mZeroPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)mIm2ColBuffer, mGemmInfo.elh[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmCuda_F32_F32_Linear_AlignCuda::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            cutlass::Status status = mGemmCudaF32F32Ln.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmCudaF32F32Ln.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }
        return NO_ERROR;
    }
    //MNN_PRINT("Conv Gemm mnk:%d-%d-%d\n", mGemmInfo.elh[0], mGemmInfo.elhPad[2], mGemmInfo.elhPad[1]);
    mGpuComputeCap = runtime->compute_capability();

    if(mGpuComputeCap < 75) {
        if(mActivationType == 1) {
            if(mFp16Infer) {
                // Create a tuple of gemm fp16 + relu kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename GemmCuda_F16_F16_Relu_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementInput_F16 *)mInputBuffer, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementOutput_F16 *)mZeroPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F16 *)mIm2ColBuffer, mGemmInfo.elh[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmCuda_F16_F16_Relu_AlignCuda::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmCudaF16F16Relu.can_implement(arguments);
                cutlass_check(status);
            
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmCudaF16F16Relu.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            } else {
                // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename GemmCuda_F16_F32_Relu_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementInput_F16 *)mInputBuffer, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementOutput_F32 *)mZeroPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)mIm2ColBuffer, mGemmInfo.elh[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmCuda_F16_F32_Relu_AlignCuda::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmCudaF16F32Relu.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmCudaF16F32Relu.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
    
        } else if(mActivationType == 2) {
    
            if(mFp16Infer) {
                // Create a tuple of gemm fp16 + relu6 kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename GemmCuda_F16_F16_Relu6_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementInput_F16 *)mInputBuffer, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementOutput_F16 *)mZeroPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F16 *)mIm2ColBuffer, mGemmInfo.elh[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmCuda_F16_F16_Relu6_AlignCuda::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmCudaF16F16Relu6.can_implement(arguments);
                cutlass_check(status);
            
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmCudaF16F16Relu6.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            } else {
                // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
                // instantiated CUTLASS kernel
                typename GemmCuda_F16_F32_Relu6_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInput_F16 *)mInputBuffer, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F32 *)mZeroPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F32 *)mIm2ColBuffer, mGemmInfo.elh[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmCuda_F16_F32_Relu6_AlignCuda::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                // Check the problem size is supported or not 
                cutlass::Status status = mGemmCudaF16F32Relu6.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmCudaF16F32Relu6.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
    
        } else {
        
            if(mFp16Infer) {
                typename GemmCuda_F16_F16_Linear_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            {(ElementInput_F16 *)mInputBuffer, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            {(ElementOutput_F16 *)mZeroPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                            {(ElementOutput_F16 *)mIm2ColBuffer, mGemmInfo.elh[2]},  //  Ptr + ldm
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmCuda_F16_F16_Linear_AlignCuda::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }

                cutlass::Status status = mGemmCudaF16F16Ln.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmCudaF16F16Ln.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            } else {
                typename GemmCuda_F16_F32_Linear_AlignCuda::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                    {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                    {(ElementInput_F16 *)mInputBuffer, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                    {(ElementOutput_F32 *)mZeroPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                    {(ElementOutput_F32 *)mIm2ColBuffer, mGemmInfo.elh[2]},  //  Ptr + ldm
                                                    {alpha, beta},          // <- tuple of alpha and beta
                                                    split_k_slices};        // <- k-dimension split factor
                size_t workspace_size = GemmCuda_F16_F32_Linear_AlignCuda::get_workspace_size(arguments);
    
                if(workspace_size != 0) {
                    workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                    mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                    mWorkspace = (void *)workspaceTensor.get()->buffer().device;
                }
    
                cutlass::Status status = mGemmCudaF16F32Ln.can_implement(arguments);
                cutlass_check(status);
    
                // Initialize CUTLASS kernel with arguments and workspace pointer
                status = mGemmCudaF16F32Ln.initialize(arguments, (uint8_t *)mWorkspace);
                cutlass_check(status);
            }
        }
    
        return NO_ERROR;
    }

    if(mActivationType == 1) {
        if(mFp16Infer) {
            // Create a tuple of gemm fp16 + relu kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmTensor_F16_F16_Relu_AlignCuda_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementInput_F16 *)mInputBuffer, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementOutput_F16 *)mZeroPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F16 *)mIm2ColBuffer, mGemmInfo.elh[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F16_Relu_AlignCuda_Sm75::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF16F16ReluSm75.can_implement(arguments);
            cutlass_check(status);
        
            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F16ReluSm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        } else {
            // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmTensor_F16_F32_Relu_AlignCuda_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            {(ElementInput_F16 *)mInputBuffer, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            {(ElementOutput_F32 *)mZeroPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                            {(ElementOutput_F32 *)mIm2ColBuffer, mGemmInfo.elh[2]},  //  Ptr + ldm
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F32_Relu_AlignCuda_Sm75::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF16F32ReluSm75.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F32ReluSm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }

    } else if(mActivationType == 2) {

        if(mFp16Infer) {
            // Create a tuple of gemm fp16 + relu6 kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmTensor_F16_F16_Relu6_AlignCuda_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            {(ElementInput_F16 *)mInputBuffer, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            {(ElementOutput_F16 *)mZeroPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                            {(ElementOutput_F16 *)mIm2ColBuffer, mGemmInfo.elh[2]},  //  Ptr + ldm
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F16_Relu6_AlignCuda_Sm75::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF16F16Relu6Sm75.can_implement(arguments);
            cutlass_check(status);
        
            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F16Relu6Sm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        } else {
            // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
            // instantiated CUTLASS kernel
            typename GemmTensor_F16_F32_Relu6_AlignCuda_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F16 *)mInputBuffer, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mZeroPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)mIm2ColBuffer, mGemmInfo.elh[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F32_Relu6_AlignCuda_Sm75::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmF16F32Relu6Sm75.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F32Relu6Sm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }

    } else {
    
        if(mFp16Infer) {
            typename GemmTensor_F16_F16_Linear_AlignCuda_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                        {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                        {(ElementInput_F16 *)mInputBuffer, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                        {(ElementOutput_F16 *)mZeroPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                        {(ElementOutput_F16 *)mIm2ColBuffer, mGemmInfo.elh[2]},  //  Ptr + ldm
                                        {alpha, beta},          // <- tuple of alpha and beta
                                        split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F16_Linear_AlignCuda_Sm75::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            cutlass::Status status = mGemmF16F16LnSm75.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F16LnSm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        } else {
            typename GemmTensor_F16_F32_Linear_AlignCuda_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                {(ElementInput_F16 *)mInputBuffer, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                {(ElementOutput_F32 *)mZeroPtr, 0},  //  Ptr + ldm  if ldm = 0, vector, 
                                                {(ElementOutput_F32 *)mIm2ColBuffer, mGemmInfo.elh[2]},  //  Ptr + ldm
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                split_k_slices};        // <- k-dimension split factor
            size_t workspace_size = GemmTensor_F16_F32_Linear_AlignCuda_Sm75::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            cutlass::Status status = mGemmF16F32LnSm75.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmF16F32LnSm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }
    }

    return NO_ERROR;
}

ErrorCode DeconvSingleInputExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    //MNN_PRINT("cuda convSingleInput onExecute in, inputsize:%d %d\n", (int)inputs.size(), workspace_size_);
    MNN_ASSERT(inputs.size() == 1);
    MNN_ASSERT(outputs.size() == 1);
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(inputs[0]);

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    const void *input_addr = (const void*)inputs[0]->deviceId();
    const void *filter_addr = mResource->mFilter;
    const void *bias_addr = mResource->mBias;
    void *output_addr = (void*)outputs[0]->deviceId();

    // Do input Rerange Pack
    {
        int maxCount = mGemmInfo.elhPad[1] * mGemmInfo.elh[2] / 4;
        int block_num = runtime->blocks_num(maxCount);
        int block_size = runtime->threads_num();

        if(mFp16Fp32MixInfer) {
            __Float22Half2<<<block_num, block_size>>>((const float*)input_addr, (half*)mInputBuffer, maxCount);
            checkKernelErrors;
        } 
    }

    // Do Gemm Compute
    if(mFp32Infer) {
        if(mActivationType == 1) {
            cutlass::Status status = mGemmCudaF32F32Relu();
            cutlass_check(status);
        } else if(mActivationType == 2) {
            cutlass::Status status = mGemmCudaF32F32Relu6();
            cutlass_check(status);
        } else {
            cutlass::Status status = mGemmCudaF32F32Ln();
            cutlass_check(status);
        }
    } else {
        if(mGpuComputeCap < 75) {
            if(mActivationType == 1) {
                if(mFp16Fp32MixInfer) {
                    cutlass::Status status = mGemmCudaF16F32Relu();
                    cutlass_check(status);
                } else {
                    cutlass::Status status = mGemmCudaF16F16Relu();
                    cutlass_check(status);
                }
            } else if(mActivationType == 2) {
                if(mFp16Fp32MixInfer) {
                    cutlass::Status status = mGemmCudaF16F32Relu6();
                    cutlass_check(status);
                } else {
                    cutlass::Status status = mGemmCudaF16F16Relu6();
                    cutlass_check(status);
                }
            } else {
                if(mFp16Fp32MixInfer) {
                    cutlass::Status status = mGemmCudaF16F32Ln();
                    cutlass_check(status);
                } else {
                    cutlass::Status status = mGemmCudaF16F16Ln();
                    cutlass_check(status);
                }
            }
        } else {
            if(mActivationType == 1) {
                if(mFp16Fp32MixInfer) {
                    cutlass::Status status = mGemmF16F32ReluSm75();
                    cutlass_check(status);
                } else {
                    cutlass::Status status = mGemmF16F16ReluSm75();
                    cutlass_check(status);
                }
            } else if(mActivationType == 2) {
                if(mFp16Fp32MixInfer) {
                    cutlass::Status status = mGemmF16F32Relu6Sm75();
                    cutlass_check(status);
                } else {
                    cutlass::Status status = mGemmF16F16Relu6Sm75();
                    cutlass_check(status);
                }
            } else {
                if(mFp16Fp32MixInfer) {
                    cutlass::Status status = mGemmF16F32LnSm75();
                    cutlass_check(status);
                } else {
                    cutlass::Status status = mGemmF16F16LnSm75();
                    cutlass_check(status);
                }
            }
        }
    }

    // Do Col2Im trans
    int height_col = mCol2ImParamter.ih;
    int width_col = mCol2ImParamter.iw;
    int num_kernels = mCol2ImParamter.ob * UP_DIV(mCol2ImParamter.oc, 8) * mCol2ImParamter.oh * mCol2ImParamter.ow * 8;

    int col2im_block_num = runtime->blocks_num(num_kernels);
    int col2im_thread_num = runtime->threads_num();

    // printf("col2im:%d, %d-%d-%d-%d-%d-%d\n %d-%d-%d-%d-%d-%d\n %d-%d\n", mCol2ImParamter.ob, mCol2ImParamter.oh, mCol2ImParamter.ow, mCol2ImParamter.oc, \
    //     mCol2ImParamter.ih, mCol2ImParamter.iw, mCol2ImParamter.ic, \
    //     mCol2ImParamter.padX, mCol2ImParamter.padY, mCol2ImParamter.kernelX, mCol2ImParamter.kernelY, mCol2ImParamter.strideX, mCol2ImParamter.strideY, \
    //     col2im_block_num, col2im_thread_num);
    
    if(mFp16Fp32MixInfer || mFp32Infer) {
        Col2Im<float><<<col2im_block_num, col2im_thread_num>>>(
            num_kernels, (const float*)mIm2ColBuffer, mCol2ImParamter.ob, mCol2ImParamter.oh, mCol2ImParamter.ow, mCol2ImParamter.oc, 
            mCol2ImParamter.kernelY, mCol2ImParamter.kernelX, mCol2ImParamter.padY, mCol2ImParamter.padX, 
            mCol2ImParamter.strideY, mCol2ImParamter.strideX, mCol2ImParamter.dilateY, mCol2ImParamter.dilateX,
            height_col, width_col, (const float*)bias_addr, (float *)output_addr);
    } else {
        Col2Im<half><<<col2im_block_num, col2im_thread_num>>>(
            num_kernels, (const half*)mIm2ColBuffer, mCol2ImParamter.ob, mCol2ImParamter.oh, mCol2ImParamter.ow, mCol2ImParamter.oc, 
            mCol2ImParamter.kernelY, mCol2ImParamter.kernelX, mCol2ImParamter.padY, mCol2ImParamter.padX, 
            mCol2ImParamter.strideY, mCol2ImParamter.strideX, mCol2ImParamter.dilateY, mCol2ImParamter.dilateX,
            height_col, width_col, (const float*)bias_addr, (half *)output_addr);
    }

    return NO_ERROR;
}

class CUDADeconvolutionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, 
            const MNN::Op* op, Backend* backend) const override {
        if (nullptr != op->main_as_Convolution2D()->quanParameter()) {
            auto quan = op->main_as_Convolution2D()->quanParameter();
            if (1 == quan->type() || 2 == quan->type()) {
                MNN_PRINT("cuda Deconv quant type 1 or 2 not support\n");
                return nullptr;
            }
        }

        if(inputs.size() == 3) {
            MNN_PRINT("Deconv inputs size:3 not support\n");
            return nullptr;
        } else if(inputs.size() == 1) {
            std::shared_ptr<DeconvSingleInputExecution::Resource> resource(new DeconvSingleInputExecution::Resource(backend, op));
            return new DeconvSingleInputExecution(backend, op, resource);
        } else {
            MNN_PRINT("Deconv inputs size:%d not support", (int)inputs.size());
            return nullptr;
        }
    }
};

CUDACreatorRegister<CUDADeconvolutionCreator> __DeConvExecution(OpType_Deconvolution);

}// namespace CUDA
}// namespace MNN

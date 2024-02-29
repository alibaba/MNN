//
//  ConvWinogradExecution.cpp
//  MNN
//
//  Created by MNN on 2022/05/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "ConvWinogradExecution.hpp"
#include "math/WingoradGenerater.hpp"
#include "WinogradTrans.cuh"

namespace MNN {
namespace CUDA {

#define UNIT 2
template<typename T>
__global__ void WinoWeightReorder(const float* GgGt, 
    T* GgGt_trans,
    const int block,
    const int co_pack,
    const int ci_pack,
    const int unitCi,
    const int unitCo
    ) {
    const int maxCount = block * co_pack * ci_pack;
    for(size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < maxCount; index += gridDim.x * blockDim.x) {
        size_t tmp =  index / ci_pack;
        size_t ci_idx =  index % ci_pack;

        size_t block_idx =  tmp / co_pack;
        size_t co_idx = tmp % co_pack;
        // [4x4, Cop, Cip, unitCi, unitCo] -->> [4x4, Cop*unitCo, Cip*unitCi]
        size_t src_idx = block_idx * (co_pack*ci_pack) + (co_idx/unitCo) * (ci_pack*unitCo) + (ci_idx/unitCi) * (unitCi*unitCo) + (ci_idx%unitCi) * unitCo + (co_idx%unitCo);
        *(GgGt_trans + index) = *(GgGt + src_idx);
    }
}

bool ConvWinogradExecution::isValid(const Convolution2D* conv) {
    //return false;
    if(conv->common()->strideX() != 1 || conv->common()->strideY() != 1) {
        return false;
    }
    if(conv->common()->dilateX() != 1 || conv->common()->dilateY() != 1) {
        return false;
    }
    if(conv->common()->padX() != 1 || conv->common()->padY() != 1) {
        return false;
    }
    return (conv->common()->kernelX() == 3) && (conv->common()->kernelY() == 3);
}

ConvWinogradExecution::Resource::Resource(Backend* backend, const MNN::Op* op) {
    mBackend = backend;
    auto runtime = static_cast<CUDABackend*>(backend)->getCUDARuntime();

    auto conv       = op->main_as_Convolution2D();
    auto common     = conv->common();
    mKernelInfo.kernelX        = common->kernelX();
    mKernelInfo.kernelY        = common->kernelY();
    mKernelInfo.groups         = common->group();
    mKernelInfo.strideX        = common->strideX();
    mKernelInfo.strideY        = common->strideY();
    mKernelInfo.dilateX        = common->dilateX();
    mKernelInfo.dilateY        = common->dilateY();

    //weight host->device
    const float* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, backend, conv, &filterDataPtr, &weightSize);
    mKernelInfo.kernelN = common->outputCount();
    mKernelInfo.kernelC = weightSize / mKernelInfo.kernelN / mKernelInfo.kernelX / mKernelInfo.kernelY;

    const int kernel = 3;
    Math::WinogradGenerater generator(UNIT, kernel, 1.0);
    std::shared_ptr<Tensor> srcWeight(Tensor::create<float>({mKernelInfo.kernelN, mKernelInfo.kernelC, mKernelInfo.kernelY, mKernelInfo.kernelX},
        (void *)filterDataPtr, Tensor::CAFFE));

    auto dstWeight = generator.allocTransformWeight(srcWeight.get(), PACK_NUMBER, PACK_NUMBER);
    generator.transformWeight(dstWeight.get(), srcWeight.get());
    auto dstWeightSize = dstWeight->elementSize();

    // Reorder weight
    {
        auto tempCacheBuffer = static_cast<CUDABackend*>(backend)->getStaticBufferPool()->alloc(dstWeightSize*sizeof(float));
        float* cacheWeight = (float*)((uint8_t*)tempCacheBuffer.first + tempCacheBuffer.second);
        runtime->memcpy(cacheWeight, dstWeight->host<uint8_t>(), dstWeightSize * sizeof(float), MNNMemcpyHostToDevice);
        if(static_cast<CUDABackend*>(backend)->getPrecision() == 1) {
            weightTensor.reset(Tensor::createDevice<int32_t>({dstWeightSize}));
        } else {
            weightTensor.reset(Tensor::createDevice<int16_t>({dstWeightSize}));
        }
        backend->onAcquireBuffer(weightTensor.get(), Backend::STATIC);
        mFilter = (void *)weightTensor.get()->buffer().device;
        auto& prop = runtime->prop();
        int cores = prop.multiProcessorCount;
        int threadNumbers = prop.maxThreadsPerBlock;

        int coPack = UP_DIV(mKernelInfo.kernelN, PACK_NUMBER) * PACK_NUMBER;
        int ciPack = UP_DIV(mKernelInfo.kernelC, PACK_NUMBER) * PACK_NUMBER;

        if(static_cast<CUDABackend*>(backend)->getPrecision() == 1) {
            WinoWeightReorder<<<cores, threadNumbers>>>((float*)cacheWeight, (float*)mFilter,
                (UNIT+kernel-1) * (UNIT+kernel-1), coPack, ciPack, PACK_NUMBER, PACK_NUMBER);
            checkKernelErrors;
        } else {
            WinoWeightReorder<<<cores, threadNumbers>>>((float*)cacheWeight, (half*)mFilter,
                    (UNIT+kernel-1) * (UNIT+kernel-1), coPack, ciPack, PACK_NUMBER, PACK_NUMBER);
            checkKernelErrors;
        }
        static_cast<CUDABackend*>(backend)->getStaticBufferPool()->free(tempCacheBuffer);
    }
    
    // Copy Bias
    int biasSize = conv->bias()->size();
    int alignSize = UP_DIV(biasSize, PACK_NUMBER) * PACK_NUMBER;
    biasTensor.reset(Tensor::createDevice<uint32_t>({alignSize}));
    backend->onAcquireBuffer(biasTensor.get(), Backend::STATIC);

    mBias = (void *)biasTensor.get()->buffer().device;
    cuda_check(cudaMemset(mBias, 0, alignSize*sizeof(float)));
    cuda_check(cudaMemcpy(mBias, conv->bias()->data(), conv->bias()->size()*sizeof(float), cudaMemcpyHostToDevice));

}

ConvWinogradExecution::Resource::~Resource() {
    // Do nothing
}

ConvWinogradExecution::ConvWinogradExecution(Backend* backend, const MNN::Op* op, std::shared_ptr<Resource> res) :
    #ifdef ENABLE_CUDA_TUNE_PARAM
    CutlassGemmTuneCommonExecution(backend),
    #else
    Execution(backend),
    #endif
    mOp(op) 
{
    mResource = res;
    int precisonLevel = static_cast<CUDABackend*>(backend)->getPrecision();
    mFp16Infer = (precisonLevel == 2);
    mFp32Infer = (precisonLevel == 1);
    mFp16Fp32MixInfer = (precisonLevel == 0);
}
ConvWinogradExecution::~ConvWinogradExecution() {
    // Nothing
}

bool ConvWinogradExecution::onClone(Backend* backend, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if(nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvWinogradExecution(backend, op, mResource);
    *dst = dstExe;
    return true;
}

ErrorCode ConvWinogradExecution::onResize(const std::vector<Tensor*>  &inputs, const std::vector<Tensor*> &outputs) {

    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();

    auto input = inputs[0];
    auto output = outputs[0];
    auto convCommon = mOp->main_as_Convolution2D()->common();
    auto pads = ConvolutionCommon::convolutionPadFull(input, output, mOp->main_as_Convolution2D()->common());
    mPadX = std::get<0>(pads);
    mPadY = std::get<1>(pads);
    int ic = input->channel();
    int icDiv = UP_DIV(ic, PACK_NUMBER);
    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(input);

    auto wUnit = UP_DIV(output->width(), UNIT);
    auto hUnit = UP_DIV(output->height(), UNIT);

    int e = wUnit * hUnit * output->batch();
    int l = ic;
    int h = output->channel();

    mGemmInfo.elh[0] = e;
    mGemmInfo.elh[1] = l;
    mGemmInfo.elh[2] = h;
    mGemmInfo.elhPad[0] = UP_DIV(e, PACK_NUMBER) * PACK_NUMBER;
    mGemmInfo.elhPad[1] = UP_DIV(l, PACK_NUMBER) * PACK_NUMBER;
    mGemmInfo.elhPad[2] = UP_DIV(h, PACK_NUMBER) * PACK_NUMBER;

    mActivationType = convCommon->relu() ? 1 : convCommon->relu6() ? 2 : 0;
    //MNN_PRINT("!!conv size:3-1, %d-%d-%d, %d-%d-%d\n", input->height(), input->width(), input->channel(), output->height(), output->width(), output->channel());

    int block = UNIT + convCommon->kernelY() - 1;
    mBlock2 = block * block;
    auto pool = static_cast<CUDABackend*>(backend())->getBufferPool();
    size_t BtdB_bytes = 2;
    if(mFp32Infer) {
        BtdB_bytes = 4;
    }
    auto bufferData = pool->alloc(BtdB_bytes * mBlock2 * mGemmInfo.elhPad[0] * mGemmInfo.elhPad[1]);
    mBtdB_Buffer = (void*)((uint8_t*)bufferData.first + bufferData.second);
    
    auto bufferMatmul = pool->alloc(bytes * mBlock2 * mGemmInfo.elh[0] * mGemmInfo.elhPad[2]);
    mMatmul_Buffer = (void*)((uint8_t*)bufferMatmul.first + bufferMatmul.second);
    
    pool->free(bufferData);
    pool->free(bufferMatmul);

    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    cutlass::gemm::GemmCoord problem_size(mGemmInfo.elh[0], mGemmInfo.elhPad[2], mGemmInfo.elhPad[1]);// m n k

    if(mFp32Infer) {
        typename GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Column::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                    {(ElementInput_F32 *)mBtdB_Buffer, mGemmInfo.elhPad[1]},  // Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                    {(ElementInput_F32 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]), // batch_stride_B
                    {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector,
                    (int64_t)(0), // batch_stride_bias
                    {(ElementOutput_F32 *)mMatmul_Buffer, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                    (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                    {alpha, beta},          // <- tuple of alpha and beta
                    mBlock2};                // batch_count

        size_t workspace_size = GemmBatchedCuda_F32_F32_Linear_AlignCuda_Row_Column::get_workspace_size(arguments);

        if(workspace_size != 0) {
            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
            mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
        }

        // Check the problem size is supported or not 
        cutlass::Status status = mGemmBatchedCudaF32F32Ln.can_implement(arguments);
        cutlass_check(status);

        // Initialize CUTLASS kernel with arguments and workspace pointer
        status = mGemmBatchedCudaF32F32Ln.initialize(arguments, (uint8_t *)mWorkspace);
        cutlass_check(status);

        return NO_ERROR;
    }

    mGpuComputeCap = runtime->compute_capability();
    //MNN_PRINT("Gpu smArch is sm_%d\n", mGpuComputeCap);

    if(mGpuComputeCap < 75) {
        if(mFp16Infer) {
            typename GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Column::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {(ElementInput_F16 *)mBtdB_Buffer, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                                {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]), // batch_stride_B
                                                {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                (int64_t)(0), // batch_stride_bias
                                                {(ElementOutput_F16 *)mMatmul_Buffer, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                mBlock2};                // batch_count
    
            size_t workspace_size = GemmBatchedCuda_F16_F16_Linear_AlignCuda_Row_Column::get_workspace_size(arguments);
    
            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }
    
            // Check the problem size is supported or not 
            cutlass::Status status = mGemmBatchedCudaF16F16Ln.can_implement(arguments);
            cutlass_check(status);
    
            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmBatchedCudaF16F16Ln.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        } else {
    
            typename GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Column::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {(ElementInput_F16 *)mBtdB_Buffer, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                                {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]), // batch_stride_B
                                                {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                (int64_t)(0), // batch_stride_bias
                                                {(ElementOutput_F32 *)mMatmul_Buffer, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                mBlock2};                // batch_count
    
            size_t workspace_size = GemmBatchedCuda_F16_F32_Linear_AlignCuda_Row_Column::get_workspace_size(arguments);
    
            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }
    
            // Check the problem size is supported or not 
            cutlass::Status status = mGemmBatchedCudaF16F32Ln.can_implement(arguments);
            cutlass_check(status);
    
            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmBatchedCudaF16F32Ln.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }
    
        return NO_ERROR;
    }
    //MNN_PRINT("Winograd BatchGemm batch:%d, MNK:%d-%d-%d\n", mBlock2, mGemmInfo.elh[0], mGemmInfo.elhPad[2], mGemmInfo.elhPad[1]);
    if(mFp16Infer) {
    #ifdef ENABLE_CUDA_TUNE_PARAM
        if(mGpuComputeCap >= 80 ) {
            mIsTuned = true;
            /*
            // 0 -> Gemm, 1~N -> BatchGemm
            int32_t batchSize = 0;
            // [0]->A, [1]->B, [2]->bias, [3]->output
            std::pair<void *, int32_t> ptrOffset[4]; 
            int32_t batchOffset[4];
            // [0]->alpha, [1]->beta, [2]->splitK
            int32_t coefs[3]; 
            // 0 -> RowColumn, 1 -> RowRow
            int32_t layout;
            bool epilogueVectorize
            */
            mInfo.problemSize[0] = mGemmInfo.elh[0];
            mInfo.problemSize[1] = mGemmInfo.elhPad[2];
            mInfo.problemSize[2] = mGemmInfo.elhPad[1];

            mInfo.coefs[0] = 1;
            mInfo.coefs[1] = 0;
            mInfo.epilogueVectorize = true;
            mInfo.epilogueType = 0;// Linear
            mInfo.precisionType = 2;// FP16_FP16
            mInfo.backend = mResource->mBackend;

            mInfo.batchSize = mBlock2;
            mInfo.layout = 0;

            mInfo.ptrOffset[0] = std::make_pair((void *)mBtdB_Buffer, mGemmInfo.elhPad[1]);
            mInfo.ptrOffset[1] = std::make_pair((void *)mResource->mFilter, mGemmInfo.elhPad[1]);
            mInfo.ptrOffset[2] = std::make_pair((void *)mResource->mBias, 0);
            mInfo.ptrOffset[3] = std::make_pair((void *)mMatmul_Buffer, mGemmInfo.elhPad[2]);

            mInfo.batchOffset[0] = mGemmInfo.elh[0] * mGemmInfo.elhPad[1];
            mInfo.batchOffset[1] = mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2];
            mInfo.batchOffset[2] = 0;
            mInfo.batchOffset[3] = mGemmInfo.elh[0] * mGemmInfo.elhPad[2];

            getGemmBatchedTensorCoreFloat16Param(&mInfo);
            // set preferd block shape argments
            setGemmBatchedTensorCoreFloat16Argments(&mInfo);
        }
    #endif
        if(!mIsTuned) {
            typename GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                                {(ElementInput_F16 *)mBtdB_Buffer, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                                {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]), // batch_stride_B
                                                {(ElementOutput_F16 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                                (int64_t)(0), // batch_stride_bias
                                                {(ElementOutput_F16 *)mMatmul_Buffer, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                                (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                                                {alpha, beta},          // <- tuple of alpha and beta
                                                mBlock2};                // batch_count

            size_t workspace_size = GemmBatchedTensor_F16_F16_Linear_AlignTensor_Row_Column_Sm75::get_workspace_size(arguments);

            if(workspace_size != 0) {
                workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
                mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
                mWorkspace = (void *)workspaceTensor.get()->buffer().device;
            }

            // Check the problem size is supported or not 
            cutlass::Status status = mGemmBatchedF16F16LnSm75.can_implement(arguments);
            cutlass_check(status);

            // Initialize CUTLASS kernel with arguments and workspace pointer
            status = mGemmBatchedF16F16LnSm75.initialize(arguments, (uint8_t *)mWorkspace);
            cutlass_check(status);
        }
    } else {

        typename GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm75::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            {(ElementInput_F16 *)mBtdB_Buffer, mGemmInfo.elhPad[1]},  // Ptr + ldm
                                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[1]), // batch_stride_A
                                            {(ElementInput_F16 *)mResource->mFilter, mGemmInfo.elhPad[1]},  //  Ptr + ldm
                                            (int64_t)(mGemmInfo.elhPad[1] * mGemmInfo.elhPad[2]), // batch_stride_B
                                            {(ElementOutput_F32 *)mResource->mBias, 0},  //  Ptr + ldm  if ldm = 0, vector,
                                            (int64_t)(0), // batch_stride_bias
                                            {(ElementOutput_F32 *)mMatmul_Buffer, mGemmInfo.elhPad[2]},  //  Ptr + ldm
                                            (int64_t)(mGemmInfo.elh[0] * mGemmInfo.elhPad[2]),  // batch_stride_C
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            mBlock2};                // batch_count

        size_t workspace_size = GemmBatchedTensor_F16_F32_Linear_AlignTensor_Row_Column_Sm75::get_workspace_size(arguments);

        if(workspace_size != 0) {
            workspaceTensor.reset(Tensor::createDevice<int8_t>({(int)workspace_size}));
            mResource->mBackend->onAcquireBuffer(workspaceTensor.get(), Backend::STATIC);
            mWorkspace = (void *)workspaceTensor.get()->buffer().device;
        }

        // Check the problem size is supported or not 
        cutlass::Status status = mGemmBatchedF16F32LnSm75.can_implement(arguments);
        cutlass_check(status);

        // Initialize CUTLASS kernel with arguments and workspace pointer
        status = mGemmBatchedF16F32LnSm75.initialize(arguments, (uint8_t *)mWorkspace);
        cutlass_check(status);
    }

    return NO_ERROR;
}

ErrorCode ConvWinogradExecution::onExecute(const std::vector<Tensor*> &inputs, const std::vector<Tensor*> &outputs) {
    auto runtime = static_cast<CUDABackend*>(backend())->getCUDARuntime();
    auto input = inputs[0];
    auto output = outputs[0];
    auto& prop = runtime->prop();
    int cores = prop.multiProcessorCount;
    int threadNumbers = prop.maxThreadsPerBlock / 2;

    int co_pack = UP_DIV(mResource->mKernelInfo.kernelN, PACK_NUMBER) * PACK_NUMBER;
    int ci_pack = UP_DIV(mResource->mKernelInfo.kernelC, PACK_NUMBER) * PACK_NUMBER;

    auto bytes = static_cast<CUDABackend*>(backend())->getBytes(input);
    const void *input_addr = (const void*)input->deviceId();
    const void *mGgGt_Buffer = mResource->mFilter;
    const void *bias_addr = mResource->mBias;
    void *output_addr = (void*)output->deviceId();

    const int kernel = 3;
    const int wUnit = UP_DIV(input->width(), UNIT);
    const int hUnit = UP_DIV(input->height(), UNIT);
    DivModFast lD(ci_pack);
    DivModFast hD(co_pack);
    DivModFast whD(wUnit * hUnit);
    DivModFast wD(wUnit);

    int total = mGemmInfo.elh[0] * ci_pack;
    int block_num = runtime->blocks_num(total);
    int block_size = runtime->threads_num();
    if(mFp32Infer) {
        WinoInputTrans<<<block_num, block_size>>>((const float*)input_addr, (float*)mBtdB_Buffer, UNIT,
                (UNIT+kernel-1)*(UNIT+kernel-1), input->channel(), ci_pack, 
                total, lD, whD, wD,
                mPadX, mPadY, input->width(), input->height());
        checkKernelErrors;
    } else if(mFp16Fp32MixInfer) {
        WinoInputTrans<<<block_num, block_size>>>((const float*)input_addr, (half*)mBtdB_Buffer, UNIT,
                (UNIT+kernel-1)*(UNIT+kernel-1), input->channel(), ci_pack, 
                total, lD, whD, wD,
                mPadX, mPadY, input->width(), input->height());
        checkKernelErrors;
    } else {
        WinoInputTrans<<<block_num, block_size>>>((const half*)input_addr, (half*)mBtdB_Buffer, UNIT,
                (UNIT+kernel-1)*(UNIT+kernel-1), input->channel(), ci_pack, 
                total, lD, whD, wD,
                mPadX, mPadY, input->width(), input->height());
        checkKernelErrors;
    }

    int iBlock = 0;

    if(mFp32Infer) {
        cutlass::Status status = mGemmBatchedCudaF32F32Ln();
        cutlass_check(status);
    } else {
        if(mGpuComputeCap < 75) {
            if (mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmBatchedCudaF16F32Ln();
                cutlass_check(status);
            } else {
                cutlass::Status status = mGemmBatchedCudaF16F16Ln();
                cutlass_check(status);
            }
        } else {
            if (mFp16Fp32MixInfer) {
                cutlass::Status status = mGemmBatchedF16F32LnSm75();
                cutlass_check(status);
            } else {
                #ifdef ENABLE_CUDA_TUNE_PARAM
                if(mIsTuned) {
                    runGemmBatchedTensorCoreFloat16Infer(&mInfo);
                }
                #endif 
                if(!mIsTuned) {
                    cutlass::Status status = mGemmBatchedF16F16LnSm75();
                    cutlass_check(status);
                }
            }
        }
    }
    int count = mGemmInfo.elh[0] * co_pack;
    block_num = runtime->blocks_num(count);
    block_size = runtime->threads_num();
    if (mFp16Fp32MixInfer || mFp32Infer) {
        WinoTrans2Output<<<block_num, block_size>>>((const float*)mMatmul_Buffer, (const float*)bias_addr, (float*)output_addr,
                UNIT, mBlock2, output->channel(), co_pack, 
                count, hD, whD, wD,
                output->width(), output->height(),
                mActivationType);
        checkKernelErrors;
    } else {
        WinoTrans2Output<<<block_num, block_size>>>((const half*)mMatmul_Buffer, (const float*)bias_addr, (half*)output_addr,
                UNIT, mBlock2, output->channel(), co_pack, 
                count, hD, whD, wD,
                output->width(), output->height(),
                mActivationType);
        checkKernelErrors;
    }

    return NO_ERROR;
}


} // namespace CUDA
} // namespace MNN

#include "backend/cpu/KleidiAIConvolutionDepthwise.hpp"

#ifdef MNN_KLEIDIAI_ENABLED

#include <string.h>
#include "core/Concurrency.h"
#include "backend/cpu/compute/Int8FunctionsOpt.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"

namespace MNN {
    template<typename T>
void nchw_to_nhwc_optimized(const T* src, T* dst, 
                            int batch, int channel, int height, int width) {
    const int hw = height * width;
    const int chw = channel * hw;
    const int wc = width * channel;
    
    for (int n = 0; n < batch; ++n) {
        const T* src_batch = src + n * chw;
        T* dst_batch = dst + n * chw;
        
        for (int c = 0; c < channel; ++c) {
            const T* src_channel = src_batch + c * hw;
            
            for (int h = 0; h < height; ++h) {
                const T* src_row = src_channel + h * width;
                T* dst_row = dst_batch + h * wc + c;
                
                for (int w = 0; w < width; ++w) {
                    dst_row[w * channel] = src_row[w];
                }
            }
        }
    }
}

KleidiAIConvolutionDepthwise::KleidiAIDepthwiseExecution::KleidiAIDepthwiseExecution(const Convolution2DCommon* common, Backend* b,
                                                        const float* originWeight, size_t originWeightSize,
                                                        const float* bias, size_t biasSize) 
    : MNN::CPUConvolution(common, b) {
    int kernel_height  = common->kernelY();
    int kernel_width   = common->kernelX();
    int channels       = common->outputCount();
    int packedRhsSize = kai_rhs_get_dst_size_dwconv_pack_x32p1vlx1b_x32_x32_sme(kernel_height, kernel_width, channels);
    mPackedRhs.reset(Tensor::createDevice<uint8_t>(std::vector<int>{packedRhsSize}));
    bool success = b->onAcquireBuffer(mPackedRhs.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Error for alloc memory for CPUConvolutionDepthwise\n");
        mValid = false;
        return;
    }
    mNumber = ((CPUBackend*)b)->threadNumber();
    mWeightTemp.reset(Tensor::createDevice<uint8_t>(std::vector<int>{channels * kernel_height * kernel_width * (int)sizeof(float)}));
    success = b->onAcquireBuffer(mWeightTemp.get(), Backend::STATIC);
    if (!success) {
        MNN_ERROR("Error for alloc memory for CPUConvolutionDepthwise\n");
        mValid = false;
        return;
    }
    auto weightTempPtr = mWeightTemp->host<float>();
    nchw_to_nhwc_optimized(originWeight, weightTempPtr, 1, channels, kernel_height, kernel_width);
    kai_run_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme(kernel_height, kernel_width, kernel_height, kernel_width, channels, weightTempPtr, bias, mPackedRhs.get()->host<void>());
    b->onReleaseBuffer(mWeightTemp.get(), Backend::STATIC);
}
    
ErrorCode KleidiAIConvolutionDepthwise::KleidiAIDepthwiseExecution::onResize(const std::vector<Tensor*>& inputs,
                                                                 const std::vector<Tensor*>& outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto input   = inputs[0];
    auto output = outputs[0];
    TensorUtils::getDescribe(&mOutputNHWC)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
    mOutputNHWC.buffer().dimensions                         = 4;
    mOutputNHWC.buffer().dim[0].extent                      = output->batch();
    mOutputNHWC.buffer().dim[1].extent                      = output->height();
    mOutputNHWC.buffer().dim[2].extent                      = output->width();
    mOutputNHWC.buffer().dim[3].extent                      = output->channel();
    mOutputNHWC.buffer().type                               = output->getType();
    auto success = backend()->onAcquireBuffer(&mOutputNHWC, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    TensorUtils::getDescribe(&mInputNHWC)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
    mInputNHWC.buffer().dimensions                         = 4;
    mInputNHWC.buffer().dim[0].extent                      = input->batch();
    mInputNHWC.buffer().dim[1].extent                      = input->height();
    mInputNHWC.buffer().dim[2].extent                      = input->width();
    mInputNHWC.buffer().dim[3].extent                      = input->channel();
    mInputNHWC.buffer().type                               = input->getType();
    success                                                = backend()->onAcquireBuffer(&mInputNHWC, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    backend()->onReleaseBuffer(&mOutputNHWC, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mInputNHWC, Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode KleidiAIConvolutionDepthwise::KleidiAIDepthwiseExecution::onExecute(const std::vector<Tensor*>& inputs,
                                                                  const std::vector<Tensor*>& outputs) {
    auto inputTensor  = inputs[0];
    auto outputTensor = outputs[0];
    const auto srcOrigin = mInputNHWC.host<uint8_t>();
    auto dstOrigin       = mOutputNHWC.host<uint8_t>();
    auto postData = getPostParameters();
    auto output_height = outputTensor->height();
    auto core       = static_cast<CPUBackend*>(backend())->functions();
    auto batch = inputTensor->batch();       
    
    MNN_CONCURRENCY_BEGIN(tId, mNumber) {
        CPUTensorConverter::convert(inputTensor, &mInputNHWC, core, tId, mNumber);
    }
    MNN_CONCURRENCY_END();

    //CPUTensorConverter::convert(inputTensor, &mInputNHWC, core);

    constexpr size_t rows_handled = 4;  // no of rows kernel handles each time.
    for(size_t b = 0; b < batch; b++)  {
        const auto srcOriginBatch = srcOrigin + b * inputTensor->height() * inputTensor->width() * inputTensor->channel() * sizeof(float);
        auto dstOriginBatch       = dstOrigin + b * outputTensor->height() * outputTensor->width() * outputTensor->channel() * sizeof(float);
        for (size_t out_row = 0; out_row < output_height; out_row += rows_handled) {
            // Variables below used to calculate start of input pointer.
            const int start_in_row = out_row - mPadY;
            const size_t pad_top = (start_in_row < 0) ? (-start_in_row) : 0;
            const size_t in_row = (start_in_row < 0) ? 0 : start_in_row;

            // Calculate row strides for pointer.
            const size_t in_row_stride_bytes = (inputTensor->width() * inputTensor->channel() * sizeof(float));
            const size_t out_row_stride_bytes = (outputTensor->width() * outputTensor->channel() * sizeof(float));

            // Number of input rows that can be read, number of output rows to calculate.
            const size_t valid_input_rows = (in_row < inputTensor->height()) ? (inputTensor->height() - in_row) : 0;
            const size_t valid_out_rows = (outputTensor->height() - out_row);

            // Increment output/input pointers according to tile being calculated.
            auto out_offset = kai_get_dst_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
                out_row, out_row_stride_bytes);
            auto in_offset = kai_get_src_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
                in_row, in_row_stride_bytes);
            const auto inptr = (uint8_t*)srcOriginBatch + in_offset;
            auto outptr = (uint8_t*)dstOriginBatch + out_offset;

            // NOTE: Kernel expects strides to be passed as bytes.
            // f32_f32_f32p1vlx1b -> f32 output, f32 LHS, packed F32 rhs (with bias) as 1VL blocks.
            // 3x3_s : 3x3 filter with stride 1
            // 4xc : 4 rows across all output channels (plane c) is produced.
            kai_run_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
                inptr, mPackedRhs.get()->host<void>(), outptr, in_row_stride_bytes, inputTensor->channel() * sizeof(float),
                out_row_stride_bytes, outputTensor->channel() * sizeof(float), valid_input_rows, valid_out_rows,
                mPadX, pad_top, 0.0f, postData[2], postData[3]);
        }
    }

    MNN_CONCURRENCY_BEGIN(tId, mNumber) {
       CPUTensorConverter::convert(&mOutputNHWC, outputTensor, core, tId, mNumber);
    }
    MNN_CONCURRENCY_END();
    //CPUTensorConverter::convert(&mOutputNHWC, outputTensor, core);
    return NO_ERROR;
}

} // namespace MNN

#endif // defined(MNN_KLEIDIAI_ENABLED)

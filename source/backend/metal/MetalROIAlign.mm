//
//  MetalROIAlign.mm
//  MNN
//

#import "MetalROIAlign.hpp"
#import "MetalBackend.hpp"
#import "MNNMetalContext.h"
#import "core/Macro.h"
#import "core/TensorUtils.hpp"

#if MNN_METAL_ENABLED
namespace MNN {

MetalROIAlign::MetalROIAlign(Backend *backend, const MNN::Op *op)
    : MetalExecution(backend) {
    auto param = op->main_as_RoiParameters();
    mPooledWidth   = param->pooledWidth();
    mPooledHeight  = param->pooledHeight();
    mSpatialScale  = param->spatialScale();
    mSamplingRatio = param->samplingRatio();
    mAligned       = param->aligned();
    mPoolType      = param->poolType();

    auto mtbn = static_cast<MetalBackend *>(backend);
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    // 12 fields: 11 int + 1 float (spatial_scale at index 8)
    mConstBuffer = [context newDeviceBuffer:12 * sizeof(int) access:CPUWriteOnly];
}

ErrorCode MetalROIAlign::onResize(const std::vector<Tensor *> &inputs,
                                   const std::vector<Tensor *> &outputs) {
    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();
    auto input   = inputs[0];
    auto output  = outputs[0];

    int inputWidth  = input->width();
    int inputHeight = input->height();
    int inputBatch  = input->batch();
    int channel     = input->channel();
    int numSlice    = UP_DIV(channel, 4);
    int numROI      = output->batch();
    bool hasBatchIndices = (inputs.size() >= 3);

    // Select pipeline based on pool type
    NSString *kernelName = (mPoolType == PoolType_MAXPOOL) ? @"roi_align_max" : @"roi_align_avg";
    mPipeline = [context pipelineWithName:kernelName fp16:backend->useFp16InsteadFp32()];

    // Fill constant buffer (matches roi_align_shape struct in shader)
    auto cst = (int *)mConstBuffer.contents;
    cst[0]  = inputWidth;                    // input_width
    cst[1]  = inputHeight;                   // input_height
    cst[2]  = inputWidth * inputHeight;      // input_size (H*W)
    cst[3]  = inputBatch;                    // input_batch
    cst[4]  = mPooledWidth;                  // output_width (pooled_width)
    cst[5]  = mPooledHeight;                 // output_height (pooled_height)
    cst[6]  = mPooledWidth * mPooledHeight;  // output_size
    cst[7]  = numROI;                        // num_roi
    ((float *)cst)[8] = mSpatialScale;       // spatial_scale (float)
    cst[9]  = mSamplingRatio;                // sampling_ratio
    cst[10] = mAligned ? 1 : 0;             // aligned
    cst[11] = hasBatchIndices ? 1 : 0;      // has_batch_indices

    // Compute threadgroup size
    // Grid: (pooledWidth, pooledHeight, numROI * numSlice)
    auto threads = MTLSizeMake(mPooledWidth, mPooledHeight, numROI * numSlice);
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:threads];

    return NO_ERROR;
}

void MetalROIAlign::onEncode(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs,
                              id<MTLComputeCommandEncoder> encoder) {
    auto input  = inputs[0];
    auto rois   = inputs[1];
    auto output = outputs[0];

    [encoder setComputePipelineState:mPipeline];
    MetalBackend::setTensor(input, encoder, 0);   // buffer(0) = input feature map (ftype4, NC4HW4)
    MetalBackend::setTensor(rois, encoder, 1);    // buffer(1) = ROI data (ftype, linear)
    MetalBackend::setTensor(output, encoder, 2);  // buffer(2) = output (ftype4, NC4HW4)
    [encoder setBuffer:mConstBuffer offset:0 atIndex:3]; // buffer(3) = constants

    // buffer(4) = batch indices (optional)
    if (inputs.size() >= 3) {
        MetalBackend::setTensor(inputs[2], encoder, 4);
    } else {
        // Set a dummy buffer when batch_indices not provided (shader won't use it)
        [encoder setBuffer:mConstBuffer offset:0 atIndex:4];
    }

    [encoder dispatchThreadgroups:mThreads.first
            threadsPerThreadgroup:mThreads.second];
}

class MetalROIAlignCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs,
                                const MNN::Op *op, Backend *backend,
                                const std::vector<Tensor *> &outputs) const override {
        auto param = op->main_as_RoiParameters();
        if (param == nullptr) {
            MNN_ERROR("MetalROIAlign: RoiParameters is null\n");
            return nullptr;
        }
        // Only support forward inference (outputGrad=false)
        if (param->outputGrad()) {
            MNN_ERROR("MetalROIAlign: outputGrad=true not supported on Metal\n");
            return nullptr;
        }
        return new MetalROIAlign(backend, op);
    }
};

REGISTER_METAL_OP_CREATOR(MetalROIAlignCreator, OpType_ROIAlign);

} // namespace MNN
#endif /* MNN_METAL_ENABLED */
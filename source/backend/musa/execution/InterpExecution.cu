#include "InterpExecution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename T>
__global__ void InterpNearestKernel(const T* src, T* dst, 
                                     int inBatch, int inChannels, 
                                     int inHeight, int inWidth,
                                     int outHeight, int outWidth,
                                     float heightScale, float widthScale) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = inBatch * inChannels * outHeight * outWidth;
    
    if (index < totalSize) {
        int tmp = index;
        int w = tmp % outWidth;
        tmp /= outWidth;
        int h = tmp % outHeight;
        tmp /= outHeight;
        int c = tmp % inChannels;
        int b = tmp / inChannels;
        
        int inX = __float2int_rd(w * widthScale);
        int inY = __float2int_rd(h * heightScale);
        
        inX = min(max(inX, 0), inWidth - 1);
        inY = min(max(inY, 0), inHeight - 1);
        
        int inIndex = ((b * inChannels + c) * inHeight + inY) * inWidth + inX;
        dst[index] = src[inIndex];
    }
}

template<typename T>
__global__ void InterpBilinearKernel(const T* src, T* dst,
                                      int inBatch, int inChannels,
                                      int inHeight, int inWidth,
                                      int outHeight, int outWidth,
                                      float heightScale, float widthScale) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = inBatch * inChannels * outHeight * outWidth;
    
    if (index < totalSize) {
        int tmp = index;
        int w = tmp % outWidth;
        tmp /= outWidth;
        int h = tmp % outHeight;
        tmp /= outHeight;
        int c = tmp % inChannels;
        int b = tmp / inChannels;
        
        float inX = (w + 0.5f) * widthScale - 0.5f;
        float inY = (h + 0.5f) * heightScale - 0.5f;
        
        int x0 = __float2int_rd(inX);
        int y0 = __float2int_rd(inY);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        
        x0 = max(0, x0);
        y0 = max(0, y0);
        x1 = min(x1, inWidth - 1);
        y1 = min(y1, inHeight - 1);
        
        float dx = inX - x0;
        float dy = inY - y0;
        
        int idx00 = ((b * inChannels + c) * inHeight + y0) * inWidth + x0;
        int idx01 = ((b * inChannels + c) * inHeight + y0) * inWidth + x1;
        int idx10 = ((b * inChannels + c) * inHeight + y1) * inWidth + x0;
        int idx11 = ((b * inChannels + c) * inHeight + y1) * inWidth + x1;
        
        float v00 = src[idx00];
        float v01 = src[idx01];
        float v10 = src[idx10];
        float v11 = src[idx11];
        
        float v0 = v00 * (1.0f - dx) + v01 * dx;
        float v1 = v10 * (1.0f - dx) + v11 * dx;
        dst[index] = v0 * (1.0f - dy) + v1 * dy;
    }
}

InterpExecution::InterpExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) 
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
    mOp = op->main_as_Interp();
}

ErrorCode InterpExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    mInBatch = input->batch();
    mInChannels = input->channel();
    mInHeight = input->height();
    mInWidth = input->width();
    
    mOutHeight = output->height();
    mOutWidth = output->width();
    
    mHeightScale = static_cast<float>(mInHeight) / mOutHeight;
    mWidthScale = static_cast<float>(mInWidth) / mOutWidth;
    
    int threads = 256;
    int blocks = (mInBatch * mInChannels * mOutHeight * mOutWidth + threads - 1) / threads;
    
    mDim3Grid = {blocks, 1, 1};
    mDim3Block = {threads, 1, 1};
    
    return NO_ERROR;
}

ErrorCode InterpExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    
    auto inputPtr = input->host<float>();
    auto outputPtr = output->host<float>();
    
    int totalSize = mInBatch * mInChannels * mOutHeight * mOutWidth;
    
    if (mOp->resizeType() == 1) { // NEAREST
        InterpNearestKernel<<<mDim3Grid, mDim3Block>>>(
            inputPtr, outputPtr,
            mInBatch, mInChannels, mInHeight, mInWidth,
            mOutHeight, mOutWidth,
            mHeightScale, mWidthScale
        );
    } else { // BILINEAR
        InterpBilinearKernel<<<mDim3Grid, mDim3Block>>>(
            inputPtr, outputPtr,
            mInBatch, mInChannels, mInHeight, mInWidth,
            mOutHeight, mOutWidth,
            mHeightScale, mWidthScale
        );
    }
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return COMPUTE_NO_SUPPORT;
    }
    
    return NO_ERROR;
}

class InterpCreator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new InterpExecution(inputs, op, backend);
    }
};

MNNCreatorRegister<InterpCreator> gInterpRegistration(OpType_Interp);

} // namespace MUSA
} // namespace MNN

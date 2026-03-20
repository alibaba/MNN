#include "GridSampleExecution.hpp"
#include "core/MusaBackend.hpp"

namespace MNN {
namespace MUSA {

template<typename T>
__global__ void GridSampleKernel(const T* input, const T* grid, T* output,
                                  int batch, int channels, int inHeight, int inWidth,
                                  int outHeight, int outWidth,
                                  bool alignCorners) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batch * channels * outHeight * outWidth;
    
    if (index < totalSize) {
        int tmp = index;
        int outW = tmp % outWidth;
        tmp /= outWidth;
        int outH = tmp % outHeight;
        tmp /= outHeight;
        int c = tmp % channels;
        int b = tmp / channels;
        
        int gridIdx = ((b * outHeight + outH) * outWidth + outW) * 2;
        float x = grid[gridIdx];
        float y = grid[gridIdx + 1];
        
        float inX, inY;
        if (alignCorners) {
            inX = (x + 1.0f) * (inWidth - 1) / 2.0f;
            inY = (y + 1.0f) * (inHeight - 1) / 2.0f;
        } else {
            inX = (x + 1.0f) * inWidth / 2.0f - 0.5f;
            inY = (y + 1.0f) * inHeight / 2.0f - 0.5f;
        }
        
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
        
        int idx00 = ((b * channels + c) * inHeight + y0) * inWidth + x0;
        int idx01 = ((b * channels + c) * inHeight + y0) * inWidth + x1;
        int idx10 = ((b * channels + c) * inHeight + y1) * inWidth + x0;
        int idx11 = ((b * channels + c) * inHeight + y1) * inWidth + x1;
        
        float v00 = input[idx00];
        float v01 = input[idx01];
        float v10 = input[idx10];
        float v11 = input[idx11];
        
        float v0 = v00 * (1.0f - dx) + v01 * dx;
        float v1 = v10 * (1.0f - dx) + v11 * dx;
        output[index] = v0 * (1.0f - dy) + v1 * dy;
    }
}

GridSampleExecution::GridSampleExecution(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend)
    : Execution(inputs, {}, backend) {
    mBackend = static_cast<MusaBackend*>(backend);
    mOp = op->main_as_GridSample();
}

ErrorCode GridSampleExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto grid = inputs[1];
    auto output = outputs[0];
    
    mBatch = input->batch();
    mChannels = input->channel();
    mInHeight = input->height();
    mInWidth = input->width();
    
    mOutHeight = grid->height();
    mOutWidth = grid->width();
    
    mAlignCorners = mOp->alignCorners();
    
    int threads = 256;
    int totalSize = mBatch * mChannels * mOutHeight * mOutWidth;
    int blocks = (totalSize + threads - 1) / threads;
    
    mDim3Grid = {blocks, 1, 1};
    mDim3Block = {threads, 1, 1};
    
    return NO_ERROR;
}

ErrorCode GridSampleExecution::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input = inputs[0];
    auto grid = inputs[1];
    auto output = outputs[0];
    
    auto inputPtr = input->host<float>();
    auto gridPtr = grid->host<float>();
    auto outputPtr = output->host<float>();
    
    GridSampleKernel<<<mDim3Grid, mDim3Block>>>(
        inputPtr, gridPtr, outputPtr,
        mBatch, mChannels, mInHeight, mInWidth,
        mOutHeight, mOutWidth,
        mAlignCorners
    );
    
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        return COMPUTE_NO_SUPPORT;
    }
    
    return NO_ERROR;
}

class GridSampleCreator : public Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend) const override {
        return new GridSampleExecution(inputs, op, backend);
    }
};

MNNCreatorRegister<GridSampleCreator> gGridSampleRegistration(OpType_GridSample);

} // namespace MUSA
} // namespace MNN

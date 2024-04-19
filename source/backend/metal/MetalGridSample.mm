//
//  MetalGridSample.mm
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "backend/metal/MetalGridSample.hpp"
#import "backend/metal/MNNMetalContext.h"

#if MNN_METAL_ENABLED
namespace MNN {

static const char* gGridSampler = R"metal(
#include <metal_stdlib>
using namespace metal;
#if GRID3D
#define CON 3
#define CODVEC float3
#define CODVECINT int3
#else
#define CON 2
#define CODVEC float2
#define CODVECINT int2
#endif

struct grid_sample_params {
    int batches;
    int channels;
    int inH;
    int inW;
    int outH;
    int outW;
    int mode;
    int paddingMode;
    int alignCorners;
    int inD;
    int outD;
};

static float getPosition(float x, int range, int alignCorners, int paddingMode) {
    if (paddingMode == 2/*GridSamplePaddingMode_REFLECTION*/) {
        // if x is on the left side of -1.0, move it to the right side of 1.0
        if (x < -1.0f) {
            x = x + ::ceil(1 - x) * 4;
        }
        // reflect
        if (x > 1.0f) {
            float l = x - 1.0f;
            int reflectionNum = ::floor(l / 2.0);
            float offset = l - reflectionNum * 2.0f;
            x = (reflectionNum % 2 == 0) ? (1 - offset) : (-1.0f + offset);
        }
    }

    float a = alignCorners ? 1.0f : 0.0f;
    float b = alignCorners ? 0.0f : 1.0f;
    return ((1 + x) * (range - a) - b) / 2.0f;
}

static int CLAMP(int v, int min, int max) {
    if ((v) < min) {
        (v) = min;
    } else if ((v) > max) {
        (v) = max;
    }
    return v;
}

#if GRID3D
static T sample(int d, int h, int w, const device T *buffer, int depth, int height, int width, int paddingMode) {
    if (h < 0 || h >= height || w < 0 || w >= width || d < 0 || d >= depth) {
        if (paddingMode == 0) {
            return 0.0f;
        }
        h = CLAMP(h, 0, height - 1);
        w = CLAMP(w, 0, width - 1);
        d = CLAMP(d, 0, depth - 1);
    }
    return buffer[d * height * width + h * width + w];
}

static T interpolate(float d, float h, float w, const device T *buffer, int depth, int height, int width, int mode,
                         int paddingMode) {
    if (mode == 1/*GridSampleMode_NEAREST*/) {
        int nd = ::floor(d+0.5f);
        int nh = ::floor(h+0.5f);
        int nw = ::floor(w+0.5f);
        return sample(nd, nh, nw, buffer, depth, height, width, paddingMode);
    }

    // mode == GridSampleMode_BILINEAR
    int w0_d = ::floor(d);
    int w0_h = ::floor(h);
    int w0_w = ::floor(w);
    int w1_d = w0_d + 1;
    int w1_h = w0_h + 1;
    int w1_w = w0_w + 1;
    T oneV = (T)((ftype)1.0f);

    T i000 = sample(w0_d, w0_h, w0_w, buffer, depth, height, width, paddingMode);
    T i001 = sample(w0_d, w0_h, w1_w, buffer, depth, height, width, paddingMode);
    T i010 = sample(w0_d, w1_h, w0_w, buffer, depth, height, width, paddingMode);
    T i011 = sample(w0_d, w1_h, w1_w, buffer, depth, height, width, paddingMode);
    T i100 = sample(w1_d, w0_h, w0_w, buffer, depth, height, width, paddingMode);
    T i101 = sample(w1_d, w0_h, w1_w, buffer, depth, height, width, paddingMode);
    T i110 = sample(w1_d, w1_h, w0_w, buffer, depth, height, width, paddingMode);
    T i111 = sample(w1_d, w1_h, w1_w, buffer, depth, height, width, paddingMode);

    
    T f0 = (T)((ftype)(w1_w - w));
    T f1 = oneV - f0;
    T h0 = (T)((ftype)(w1_h - h));
    T h1 = oneV - h0;
    T d0 = (T)((ftype)(w1_d - d));
    T d1 = oneV - d0;

    T i00 = i000 * f0 + i001 * f1;
    T i01 = i010 * f0 + i011 * f1;
    T i10 = i100 * f0 + i101 * f1;
    T i11 = i110 * f0 + i111 * f1;
    T i0 = i00 * h0 + i01 * h1;
    T i1 = i10 * h0 + i11 * h1;

    return i0 * d0 + i1 * d1;
}
#else
static T sample(int h, int w, const device T *buffer, int height, int width, int paddingMode) {
    if (h < 0 || h >= height || w < 0 || w >= width) {
        if (paddingMode == 0/*GridSamplePaddingMode_ZEROS*/) {
            return 0.0f;
        }
        // Clearly, CLAMP is the right way to go for GridSamplePaddingMode_BORDER
        // For GridSamplePaddingMode_REFLECTION, since we have reflected the values into (-1, 1),
        // the leftover reflections degrade to GridSamplePaddingMode_BORDER
        h = CLAMP(h, 0, height - 1);
        w = CLAMP(w, 0, width - 1);
    }

    return buffer[h * width + w];
}

static T interpolate(float h, float w, const device T *buffer, int height, int width, int mode,
                         int paddingMode) {
    if (mode == 1/*GridSampleMode_NEAREST*/) {
        int nh = ::floor(h+0.5f);
        int nw = ::floor(w+0.5f);
        return sample(nh, nw, buffer, height, width, paddingMode);
    }

    // mode == GridSampleMode_BILINEAR
    int w0_h = ::floor(h);
    int w0_w = ::floor(w);
    int w1_h = w0_h + 1;
    int w1_w = w0_w + 1;
    T oneV = (T)((ftype)1.0f);

    T i00 = sample(w0_h, w0_w, buffer, height, width, paddingMode);
    T i01 = sample(w0_h, w1_w, buffer, height, width, paddingMode);
    T i10 = sample(w1_h, w0_w, buffer, height, width, paddingMode);
    T i11 = sample(w1_h, w1_w, buffer, height, width, paddingMode);

    
    T f0 = (T)((ftype)(w1_w - w));
    T f1 = oneV - f0;
    T h0 = (T)((ftype)(w1_h - h));
    T h1 = oneV - h0;

    T i0 = i00 * f0 + i01 * f1;
    T i1 = i10 * f0 + i11 * f1;

    return i0 * h0 + i1 * h1;
}
#endif

kernel void main0(const device T *input [[buffer(0)]],
                   const device ftype *grid [[buffer(1)]],
                   device T *output [[buffer(2)]],
                   constant grid_sample_params &p [[buffer(3)]],
                   uint3 gid                        [[thread_position_in_grid]]) {
    if ((int)gid.x >= p.outW || (int)gid.y >= p.outH * p.outD || (int)gid.z >= p.batches)
        return;

    int gridPos = gid.z*p.outH*p.outW*CON + gid.y*p.outW*CON + gid.x*CON;
    auto x = getPosition(grid[gridPos+0], p.inW, p.alignCorners, p.paddingMode);
    auto y = getPosition(grid[gridPos+1], p.inH, p.alignCorners, p.paddingMode);
#if GRID3D
    auto z = getPosition(grid[gridPos+2], p.inD, p.alignCorners, p.paddingMode);
#endif
    
    const int channelC4 = (p.channels + 3) / 4;
    for (int c = 0; c < channelC4; ++ c) {
        auto outputPos = gid.z*channelC4*p.outH*p.outW + c*p.outH*p.outW + gid.y*p.outW + gid.x;
        auto inputPtr = input + gid.z*channelC4*p.inH*p.inW + c*p.inH*p.inW;
#if GRID3D
        output[outputPos] = interpolate(z, y, x, inputPtr, p.inD, p.inH, p.inW, p.mode, p.paddingMode);
#else
        output[outputPos] = interpolate(y, x, inputPtr, p.inH, p.inW, p.mode, p.paddingMode);
#endif
    }
}

)metal";

static id<MTLComputePipelineState> _createPipeline(MetalBackend* mtbn, const GridSample *gridSample, Tensor* outputTensor) {
    int dims = outputTensor->dimensions();
    std::string T;
    std::string ftype;
    if (mtbn->useFp16InsteadFp32()) {
        T = "half4";
        ftype = "half";
    } else {
        T = "float4";
        ftype = "float";
    }
    std::string grid3d;
    if (dims == 5) {
        grid3d = "1";
    } else {
        grid3d = "0";
    }
    std::vector<std::string> keys = {
        T,
        grid3d,
        "gridsampler",
    };
    auto pipeline = mtbn->runtime()->findPipeline(keys);
    if (nil == pipeline) {
        MTLCompileOptions *compileOptions = [[MTLCompileOptions alloc] init];
        compileOptions.preprocessorMacros = @{
            @"T" : @(T.c_str()),
            @"GRID3D": @(grid3d.c_str()),
            @"ftype":@(ftype.c_str()),
        };
        pipeline = mtbn->makeComputePipelineWithSourceOption(gGridSampler, "main0", compileOptions);
        mtbn->runtime()->insertPipeline(keys, pipeline);
    }
    return pipeline;
}
MetalGridSample::MetalGridSample(Backend *backend, const GridSample *gridSample, id<MTLComputePipelineState> pipeline)
        : MetalExecution(backend) {
    mMode = gridSample->mode();
    mPaddingMode = gridSample->paddingMode();
    mAlignCorners = gridSample->alignCorners();

    auto mtbn = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)mtbn->context();
    mParams = [context newDeviceBuffer:11*sizeof(int) access:CPUWriteOnly];
    mPipeline = pipeline;
}

ErrorCode MetalGridSample::onResize(const std::vector<Tensor *> &inputs,
                                    const std::vector<Tensor *> &outputs) {
    auto inputTensor = inputs[0];
    auto outputTensor = outputs[0];
    int dims = outputTensor->dimensions();

    ((int *)mParams.contents)[0] = inputTensor->length(0);//inputTensor->buffer().dim[0].extent; // batches
    ((int *)mParams.contents)[1] = inputTensor->length(1);//->buffer().dim[1].extent; // channels
    ((int *)mParams.contents)[2] = inputTensor->length(dims-2);//buffer().dim[2].extent; // inH
    ((int *)mParams.contents)[3] = inputTensor->length(dims-1);//buffer().dim[3].extent; // inW
    ((int *)mParams.contents)[4] = outputTensor->length(dims-2);//->buffer().dim[2].extent; // outH
    ((int *)mParams.contents)[5] = outputTensor->length(dims-1);//->buffer().dim[3].extent; // outW
    ((int *)mParams.contents)[6] = mMode;
    ((int *)mParams.contents)[7] = mPaddingMode;
    ((int *)mParams.contents)[8] = mAlignCorners;
    if (outputTensor->dimensions() == 5) {
        ((int *)mParams.contents)[9] = inputTensor->length(dims-3);
        ((int *)mParams.contents)[10] = outputTensor->length(dims-3);
    } else {
        ((int *)mParams.contents)[9] = 1;
        ((int *)mParams.contents)[10] = 1;
    }

    auto backend = static_cast<MetalBackend *>(this->backend());
    auto context = (__bridge MNNMetalContext *)backend->context();

    int batches = ((int *)mParams.contents)[0];
    int channels = ((int *)mParams.contents)[1];
    int outH = ((int *)mParams.contents)[4];
    int outW = ((int *)mParams.contents)[5];
    int outD = ((int *)mParams.contents)[10];
    mThreads = [context computeBestGroupAndLocal:mPipeline threads:MTLSizeMake(outW, outH * outD, batches)];

    //printf("re:%d %d %d, %d %d %d, %d %d\n", mThreads.first.width, mThreads.first.height, mThreads.first.depth, mThreads.second.width, mThreads.second.height, mThreads.second.depth, ((int *)mParams.contents)[3], ((int *)mParams.contents)[2]);
    return NO_ERROR;
}

void MetalGridSample::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, id<MTLComputeCommandEncoder> encoder) {
    [encoder setComputePipelineState:mPipeline];
    MetalBackend::setTensor(inputs[0], encoder, 0);
    MetalBackend::setTensor(inputs[1], encoder, 1);
    MetalBackend::setTensor(outputs[0], encoder, 2);
    [encoder setBuffer:mParams offset:0 atIndex:3];
    [encoder dispatchThreadgroups:mThreads.first threadsPerThreadgroup:mThreads.second];
}

class MetalGridSampleCreator : public MetalBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const MNN::Op *op,
                                Backend *backend, const std::vector<Tensor *>& outputs) const override {
        auto pipeline = _createPipeline(static_cast<MetalBackend*>(backend), op->main_as_GridSample(), outputs[0]);
        if (nil == pipeline) {
            return nullptr;
        }
        return new MetalGridSample(backend, op->main_as_GridSample(), pipeline);
    }
};

REGISTER_METAL_OP_CREATOR(MetalGridSampleCreator, OpType_GridSample);
} // namespace MNN
#endif /* MNN_METAL_ENABLED */

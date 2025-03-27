#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wunused-variable"

#include <metal_stdlib>
#include <simd/simd.h>
#include <metal_atomic>

using namespace metal;

struct gridSampleBuffer
{
    int4 inShape;
    int4 outShape;
    uint alignCorners;
};

struct sourceBuffer0
{
    int data[1];
};

struct sourceBuffer1
{
    float data[1];
};

struct destBuffer
{
    float data[1];
};

constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(256u, 1u, 1u);

static inline __attribute__((always_inline))
void WriteSample(thread int& positionX, thread int& positionY, thread const int& c, thread const int& n, thread const float& value_f, constant gridSampleBuffer& uGridSampleParam, device sourceBuffer0& uInput)
{
    int value = int(value_f * 16777216.0);
    int width = uGridSampleParam.inShape.x;
    int height = uGridSampleParam.inShape.y;
    positionX = clamp(positionX, 0, width - 1);
    positionY = clamp(positionY, 0, height - 1);
    int _77 = atomic_fetch_add_explicit((device atomic_int*)&uInput.data[(((0 + (positionX * uGridSampleParam.inShape.z)) + ((positionY * width) * uGridSampleParam.inShape.z)) + (((n * width) * height) * uGridSampleParam.inShape.z)) + c], value, memory_order_relaxed);
}

kernel void main0(const device destBuffer& uOutput [[buffer(0)]], device sourceBuffer0& uInput [[buffer(1)]], const device sourceBuffer1& uGrid [[buffer(2)]], constant gridSampleBuffer& uGridSampleParam [[buffer(3)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    int pos = int(gl_GlobalInvocationID.x);
    int4 inputShape = uGridSampleParam.inShape;
    int4 outputShape = uGridSampleParam.outShape;
    int total = ((outputShape.x * outputShape.y) * outputShape.z) * outputShape.w;
    if (pos < total)
    {
        int x = pos % outputShape.x;
        int tmp = pos / outputShape.x;
        int y = tmp % outputShape.y;
        tmp /= outputShape.y;
        int z = tmp % outputShape.z;
        int n = tmp / outputShape.z;
        int gridPosition = (((n * outputShape.x) * outputShape.y) + (y * outputShape.x)) + x;
        float gridX = uGrid.data[(inputShape.w * gridPosition) + 0];
        float gridY = uGrid.data[(inputShape.w * gridPosition) + 1];
        float value = uOutput.data[(((0 + (x * outputShape.z)) + ((y * outputShape.x) * outputShape.z)) + z) + (((n * outputShape.x) * outputShape.y) * outputShape.z)];
        float cordH = (gridY * float(inputShape.y)) - 0.5;
        float cordW = (gridX * float(inputShape.x)) - 0.5;
        int w0_h = int(floor(cordH));
        int w0_w = int(floor(cordW));
        int w1_h = w0_h + 1;
        int w1_w = w0_w + 1;
        float f0 = float(w1_w) - cordW;
        float f1 = 1.0 - f0;
        float h0 = float(w1_h) - cordH;
        float h1 = 1.0 - h0;
        float f00 = (f0 * h0) * value;
        float f01 = (f1 * h0) * value;
        float f10 = (f0 * h1) * value;
        float f11 = (f1 * h1) * value;
        int param = w0_w;
        int param_1 = w0_h;
        int param_2 = z;
        int param_3 = n;
        float param_4 = f00;
        WriteSample(param, param_1, param_2, param_3, param_4, uGridSampleParam, uInput);
        int param_5 = w1_w;
        int param_6 = w0_h;
        int param_7 = z;
        int param_8 = n;
        float param_9 = f01;
        WriteSample(param_5, param_6, param_7, param_8, param_9, uGridSampleParam, uInput);
        int param_10 = w0_w;
        int param_11 = w1_h;
        int param_12 = z;
        int param_13 = n;
        float param_14 = f10;
        WriteSample(param_10, param_11, param_12, param_13, param_14, uGridSampleParam, uInput);
        int param_15 = w1_w;
        int param_16 = w1_h;
        int param_17 = z;
        int param_18 = n;
        float param_19 = f11;
        WriteSample(param_15, param_16, param_17, param_18, param_19, uGridSampleParam, uInput);
    }
}


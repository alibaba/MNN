#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct gridSampleBuffer
{
    int4 inShape;
    int4 outShape;
    uint alignCorners;
};

struct sourceBuffer0
{
    float data[1];
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
float LoadSample(thread int& positionX, thread int& positionY, thread const int& c, thread const int& n, constant gridSampleBuffer& uGridSampleParam, const device sourceBuffer0& uInput)
{
    int width = uGridSampleParam.inShape.x;
    int height = uGridSampleParam.inShape.y;
    positionX = clamp(positionX, 0, width - 1);
    positionY = clamp(positionY, 0, height - 1);
    float value = uInput.data[(((0 + (positionX * uGridSampleParam.inShape.z)) + ((positionY * width) * uGridSampleParam.inShape.z)) + (((n * width) * height) * uGridSampleParam.inShape.z)) + c];
    return value;
}

kernel void main0(device destBuffer& uOutput [[buffer(0)]], const device sourceBuffer0& uInput [[buffer(1)]], const device sourceBuffer1& uGrid [[buffer(2)]], constant gridSampleBuffer& uGridSampleParam [[buffer(3)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
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
        float cordH = (gridY * float(inputShape.y)) - 0.5;
        float cordW = (gridX * float(inputShape.x)) - 0.5;
        int w0_h = int(floor(cordH));
        int w0_w = int(floor(cordW));
        int w1_h = w0_h + 1;
        int w1_w = w0_w + 1;
        float oneV = 1.0;
        int param = w0_w;
        int param_1 = w0_h;
        int param_2 = z;
        int param_3 = n;
        float _215 = LoadSample(param, param_1, param_2, param_3, uGridSampleParam, uInput);
        float i00 = _215;
        int param_4 = w1_w;
        int param_5 = w0_h;
        int param_6 = z;
        int param_7 = n;
        float _225 = LoadSample(param_4, param_5, param_6, param_7, uGridSampleParam, uInput);
        float i01 = _225;
        int param_8 = w0_w;
        int param_9 = w1_h;
        int param_10 = z;
        int param_11 = n;
        float _235 = LoadSample(param_8, param_9, param_10, param_11, uGridSampleParam, uInput);
        float i10 = _235;
        int param_12 = w1_w;
        int param_13 = w1_h;
        int param_14 = z;
        int param_15 = n;
        float _245 = LoadSample(param_12, param_13, param_14, param_15, uGridSampleParam, uInput);
        float i11 = _245;
        float f0 = float(w1_w) - cordW;
        float f1 = oneV - f0;
        float h0 = float(w1_h) - cordH;
        float h1 = oneV - h0;
        float i0 = (i00 * f0) + (i01 * f1);
        float i1 = (i10 * f0) + (i11 * f1);
        float value = (i0 * h0) + (i1 * h1);
        uOutput.data[(((0 + (x * outputShape.z)) + ((y * outputShape.x) * outputShape.z)) + z) + (((n * outputShape.x) * outputShape.y) * outputShape.z)] = value;
    }
}


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
void indexCubeMap(thread const float3& d, thread int& face, thread float& s, thread float& t)
{
    float3 absd;
    absd.x = abs(d.x);
    absd.y = abs(d.y);
    absd.z = abs(d.z);
    face = -1;
    bool _50 = absd.x >= absd.y;
    bool _58;
    if (_50)
    {
        _58 = absd.x >= absd.z;
    }
    else
    {
        _58 = _50;
    }
    float sc;
    float tc;
    float ma;
    if (_58)
    {
        if (d.x > 0.0)
        {
            face = 0;
            sc = -d.z;
            tc = -d.y;
            ma = absd.x;
        }
        else
        {
            face = 1;
            sc = d.z;
            tc = -d.y;
            ma = absd.x;
        }
    }
    bool _92 = absd.y >= absd.x;
    bool _100;
    if (_92)
    {
        _100 = absd.y >= absd.z;
    }
    else
    {
        _100 = _92;
    }
    if (_100)
    {
        if (d.y > 0.0)
        {
            face = 2;
            sc = d.x;
            tc = d.z;
            ma = absd.y;
        }
        else
        {
            face = 3;
            sc = d.x;
            tc = -d.z;
            ma = absd.y;
        }
    }
    bool _128 = absd.z >= absd.x;
    bool _136;
    if (_128)
    {
        _136 = absd.z >= absd.y;
    }
    else
    {
        _136 = _128;
    }
    if (_136)
    {
        if (d.z > 0.0)
        {
            face = 4;
            sc = d.x;
            tc = -d.y;
            ma = absd.z;
        }
        else
        {
            face = 5;
            sc = -d.x;
            tc = -d.y;
            ma = absd.z;
        }
    }
    if (ma == 0.0)
    {
        s = 0.0;
        t = 0.0;
        face = -1;
    }
    else
    {
        s = ((sc / ma) + 1.0) * 0.5;
        t = ((tc / ma) + 1.0) * 0.5;
    }
}

static inline __attribute__((always_inline))
void WriteSample(thread int& positionX, thread int& positionY, thread const int& c, thread const int& n, thread const float& value_f, constant gridSampleBuffer& uGridSampleParam, device sourceBuffer0& uInput)
{
    int value = int(value_f * 16777216.0);
    int width = uGridSampleParam.inShape.x;
    int height = uGridSampleParam.inShape.y;
    positionX = clamp(positionX, 0, width - 1);
    positionY = clamp(positionY, 0, height - 1);
    int _232 = atomic_fetch_add_explicit((device atomic_int*)&uInput.data[(((0 + (positionX * uGridSampleParam.inShape.z)) + ((positionY * width) * uGridSampleParam.inShape.z)) + (((n * width) * height) * uGridSampleParam.inShape.z)) + c], value, memory_order_relaxed);
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
        int on = tmp / outputShape.z;
        int gridPosition = (((on * outputShape.x) * outputShape.y) + (y * outputShape.x)) + x;
        float u = uGrid.data[(inputShape.w * gridPosition) + 0];
        float v = uGrid.data[(inputShape.w * gridPosition) + 1];
        float w = uGrid.data[(inputShape.w * gridPosition) + 2];
        float3 param = float3(u, v, w);
        int param_1;
        float param_2;
        float param_3;
        indexCubeMap(param, param_1, param_2, param_3);
        int face = param_1;
        float gridX = param_2;
        float gridY = param_3;
        float value = uOutput.data[(((0 + (x * outputShape.z)) + ((y * outputShape.x) * outputShape.z)) + z) + (((on * outputShape.x) * outputShape.y) * outputShape.z)];
        if (face >= 0)
        {
            int n = (on * 6) + face;
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
            int param_4 = w0_w;
            int param_5 = w0_h;
            int param_6 = z;
            int param_7 = n;
            float param_8 = f00;
            WriteSample(param_4, param_5, param_6, param_7, param_8, uGridSampleParam, uInput);
            int param_9 = w1_w;
            int param_10 = w0_h;
            int param_11 = z;
            int param_12 = n;
            float param_13 = f01;
            WriteSample(param_9, param_10, param_11, param_12, param_13, uGridSampleParam, uInput);
            int param_14 = w0_w;
            int param_15 = w1_h;
            int param_16 = z;
            int param_17 = n;
            float param_18 = f10;
            WriteSample(param_14, param_15, param_16, param_17, param_18, uGridSampleParam, uInput);
            int param_19 = w1_w;
            int param_20 = w1_h;
            int param_21 = z;
            int param_22 = n;
            float param_23 = f11;
            WriteSample(param_19, param_20, param_21, param_22, param_23, uGridSampleParam, uInput);
        }
    }
}


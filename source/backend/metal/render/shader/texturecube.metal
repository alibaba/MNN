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
void indexCubeMap(thread const float3& d, thread int& face, thread float& s, thread float& t)
{
    float3 absd;
    absd.x = abs(d.x);
    absd.y = abs(d.y);
    absd.z = abs(d.z);
    face = -1;
    bool _49 = absd.x >= absd.y;
    bool _57;
    if (_49)
    {
        _57 = absd.x >= absd.z;
    }
    else
    {
        _57 = _49;
    }
    float sc;
    float tc;
    float ma;
    if (_57)
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
    bool _91 = absd.y >= absd.x;
    bool _99;
    if (_91)
    {
        _99 = absd.y >= absd.z;
    }
    else
    {
        _99 = _91;
    }
    if (_99)
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
    bool _127 = absd.z >= absd.x;
    bool _135;
    if (_127)
    {
        _135 = absd.z >= absd.y;
    }
    else
    {
        _135 = _127;
    }
    if (_135)
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
        float value = 0.0;
        if (face >= 0)
        {
            int n = (on * 6) + face;
            float cordH = (gridY * float(inputShape.y)) - 0.5;
            float cordW = (gridX * float(inputShape.x)) - 0.5;
            int w0_h = int(floor(cordH));
            int w0_w = int(floor(cordW));
            int w1_h = w0_h + 1;
            int w1_w = w0_w + 1;
            float oneV = 1.0;
            int param_4 = w0_w;
            int param_5 = w0_h;
            int param_6 = z;
            int param_7 = n;
            float _401 = LoadSample(param_4, param_5, param_6, param_7, uGridSampleParam, uInput);
            float i00 = _401;
            int param_8 = w1_w;
            int param_9 = w0_h;
            int param_10 = z;
            int param_11 = n;
            float _411 = LoadSample(param_8, param_9, param_10, param_11, uGridSampleParam, uInput);
            float i01 = _411;
            int param_12 = w0_w;
            int param_13 = w1_h;
            int param_14 = z;
            int param_15 = n;
            float _421 = LoadSample(param_12, param_13, param_14, param_15, uGridSampleParam, uInput);
            float i10 = _421;
            int param_16 = w1_w;
            int param_17 = w1_h;
            int param_18 = z;
            int param_19 = n;
            float _431 = LoadSample(param_16, param_17, param_18, param_19, uGridSampleParam, uInput);
            float i11 = _431;
            float f0 = float(w1_w) - cordW;
            float f1 = oneV - f0;
            float h0 = float(w1_h) - cordH;
            float h1 = oneV - h0;
            float i0 = (i00 * f0) + (i01 * f1);
            float i1 = (i10 * f0) + (i11 * f1);
            value = (i0 * h0) + (i1 * h1);
        }
        uOutput.data[(((0 + (x * outputShape.z)) + ((y * outputShape.x) * outputShape.z)) + z) + (((on * outputShape.x) * outputShape.y) * outputShape.z)] = value;
    }
}


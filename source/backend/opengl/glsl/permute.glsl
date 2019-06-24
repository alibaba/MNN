
layout(binding = 0) readonly buffer srcBuffer{
    float data[];
}uInput;

layout(binding = 1) writeonly buffer dstBuffer{
    float data[];
}uOutput;

layout(location=2) uniform ivec4 dims;
layout(location=3) uniform ivec4 inImSize;
layout(location=4) uniform ivec4 outImSize;

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 inImgSize = ivec3(inImSize.xyz);
    ivec3 outImgSize = ivec3(outImSize.xyz);
    // input, output all are NCHW layout
    ivec4 dimParam = dims.xyzw;
    if(pos.x < outImgSize.x && pos.y < outImgSize.y)
    {
        int dimIndex[4];
        
        dimIndex[dimParam.y] = pos.z;
        dimIndex[dimParam.z] = pos.y;
        dimIndex[dimParam.w] = pos.x;
        int inputIndex = dimIndex[1] * inImgSize.x * inImgSize.y + dimIndex[2] * inImgSize.x + dimIndex[3];
        int outputIndex = pos.x + pos.y * outImgSize.x + pos.z * outImgSize.x * outImgSize.y;
        uOutput.data[outputIndex] = uInput.data[inputIndex];
    }
}

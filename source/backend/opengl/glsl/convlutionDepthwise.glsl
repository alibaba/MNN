layout(std430) buffer;
layout(FORMAT, binding=0) writeonly uniform mediump image3D uOutput;
layout(location=1) uniform mediump sampler3D uInput;
layout(location=2) uniform mediump sampler3D uKernel;

layout(binding=3) readonly buffer bias{
    vec4 data[];
} uBias;

layout(location=4) uniform ivec2 uPad;
layout(location=5) uniform ivec2 uKernelSize;
layout(location=6) uniform ivec2 uStride;
layout(location=7) uniform ivec2 uDilate;
// layout(location=8) uniform ivec2 uOffset;
// layout(location=9) uniform float uReluRate;
layout(location=10) uniform ivec3 uOutputSize;
layout(location=11) uniform ivec3 uInputSize;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID)*ivec3(1, 1, 1);
    ivec3 outputSize = uOutputSize;
    if (all(lessThan(pos, outputSize)))
    {
        int KSIZE_Y = uKernelSize.y;
        int KSIZE_X = uKernelSize.x;
        ivec3 inputSize = uInputSize;
        ivec2 s0 = pos.xy*uStride-uPad;
        int fx, fy, fz;
        ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, uDilate)));
        ivec2 efxy = min(uKernelSize, UP_DIV(inputSize.xy-s0, uDilate));
        vec4 color = uBias.data[pos.z];
        for (fy=sfxy.y; fy<efxy.y; ++fy)
        {
            int sy = fy*uDilate.y + s0.y;
            for (fx=sfxy.x; fx<efxy.x; ++fx)
            {
                int sx1 = fx*uDilate.x + s0.x;
                vec4 k = texelFetch(uKernel, ivec3(pos.z, fx, fy), 0);

                color  += k*texelFetch(uInput, ivec3(sx1, sy, pos.z), 0);
            }
        }
#ifdef RELU
        color = max(color, vec4(0));
#endif
#ifdef RELU6
        color = clamp(color, vec4(0), vec4(6));
#endif
        imageStore(uOutput, pos, color);
    }

}

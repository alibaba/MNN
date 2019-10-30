layout(std430) buffer;
layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(location=1) uniform mediump sampler3D uInput;
layout(location=2) uniform mediump sampler3D uKernel;

layout(binding=3) readonly buffer bias{
    vec4 data[];
} uBias;

layout(location=8) uniform int uUnroll;

layout(location=10) uniform ivec3 uOutputSize;
layout(location=11) uniform ivec3 uInputSize;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

void main()
{
    ivec3 outputSize = uOutputSize;
    if (all(lessThan(ivec3(gl_GlobalInvocationID), outputSize)))
    {
        ivec3 pos = ivec3(gl_GlobalInvocationID)*ivec3(uUnroll, 1, 1);
        ivec3 inputSize = uInputSize;
        int sy = pos.y;
        int sx = pos.x;
        int fx, fy, fz;
        vec4 color = uBias.data[pos.z];
        vec4 color2 = color;
        vec4 color3 = color;
        vec4 color4 = color;
        int kernelY = pos.z;
        for (fz=0; fz<inputSize.z; ++fz)
        {
            int kernelX = 4*fz;
            vec4 k0 = texelFetch(uKernel, ivec3(kernelX+0, kernelY, 0), 0);
            vec4 k1 = texelFetch(uKernel, ivec3(kernelX+1, kernelY, 0), 0);
            vec4 k2 = texelFetch(uKernel, ivec3(kernelX+2, kernelY, 0), 0);
            vec4 k3 = texelFetch(uKernel, ivec3(kernelX+3, kernelY, 0), 0);
            
            mat4 k = mat4(k0, k1, k2, k3);
            
            color  += k*texelFetch(uInput, ivec3(sx+0, sy, fz), 0);
            color2 += k*texelFetch(uInput, ivec3(sx+1, sy, fz), 0);
            color3 += k*texelFetch(uInput, ivec3(sx+2, sy, fz), 0);
            color4 += k*texelFetch(uInput, ivec3(sx+3, sy, fz), 0);
        }
        #ifdef RELU
        color = max(color, vec4(0));
        color2 = max(color2, vec4(0));
        color3 = max(color3, vec4(0));
        color4 = max(color4, vec4(0));
        #endif
        #ifdef RELU6
        color = clamp(color, vec4(0), vec4(6));
        color2 = clamp(color2, vec4(0), vec4(6));
        color3 = clamp(color3, vec4(0), vec4(6));
        color4 = clamp(color4, vec4(0), vec4(6));
        #endif
        imageStore(uOutput, ivec3(pos.x+0, pos.y, pos.z), color);
        imageStore(uOutput, ivec3(pos.x+1, pos.y, pos.z), color2);
        imageStore(uOutput, ivec3(pos.x+2, pos.y, pos.z), color3);
        imageStore(uOutput, ivec3(pos.x+3, pos.y, pos.z), color4);
    }
    
}

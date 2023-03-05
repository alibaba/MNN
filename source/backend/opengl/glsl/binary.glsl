
layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(location=1) uniform mediump sampler3D uInput0;
layout(location=2) uniform mediump sampler3D uInput1;
layout(location=3) uniform ivec4 imgSize;
layout(location=4) uniform int activationType;

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 inSize = imgSize.xyz;
    if(all(lessThan(pos, inSize)))
    {
#ifdef ADD
        vec4 sum = texelFetch(uInput0, pos, 0) + texelFetch(uInput1, pos, 0);
#endif
#ifdef MUL
        vec4 sum = texelFetch(uInput0, pos, 0) * texelFetch(uInput1, pos, 0);
#endif
#ifdef SUB
        vec4 sum = texelFetch(uInput0, pos, 0) - texelFetch(uInput1, pos, 0);
#endif

#ifdef REALDIV
        vec4 sum = texelFetch(uInput0, pos, 0) / texelFetch(uInput1, pos, 0);
#endif
        if(activationType == 1) {
            sum = max(sum, vec4(0));
        }
        imageStore(uOutput, pos, sum);
    }
}

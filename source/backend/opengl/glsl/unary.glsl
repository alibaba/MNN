
layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(location=1) uniform mediump sampler3D uInput0;
layout(location=3) uniform ivec4 imgSize;

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 inSize = imgSize.xyz;
    if(all(lessThan(pos, inSize)))
    {
        vec4 data = texelFetch(uInput0, pos, 0);
#ifdef EXP
        vec4 sum = exp(data);
#endif
        imageStore(uOutput, pos, sum);
    }
}

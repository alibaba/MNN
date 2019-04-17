layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(FORMAT, binding=1) readonly uniform PRECISION image3D uInput;


layout(location = 2) uniform ivec3 uSourceOffset;
layout(location = 3) uniform ivec3 uDestOffset;
layout(location = 4) uniform ivec3 uBlitSize;


layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (all(lessThan(pos, uBlitSize)))
    {
        ivec3 dstP = uDestOffset + pos;
        ivec3 srcP = uSourceOffset + pos;
        imageStore(uOutput, dstP, imageLoad(uInput, srcP));
    }
}

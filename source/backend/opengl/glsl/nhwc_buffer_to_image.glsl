layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uImage;

layout(binding=1) readonly buffer destBuffer{
    float data[];
} uInBuffer;

layout(location = 2) uniform int uWidth;
layout(location = 3) uniform int uHeight;
layout(location = 4) uniform int uChannel;

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (pos.x < uWidth && pos.y < uHeight)
    {
        vec4 color;
        int z = pos.z*4;

        color.r = uInBuffer.data[pos.y*uWidth*uChannel + pos.x*uChannel + (z+0)];
        color.g = uInBuffer.data[pos.y*uWidth*uChannel + pos.x*uChannel + (z+1)];
        color.b = uInBuffer.data[pos.y*uWidth*uChannel + pos.x*uChannel + (z+2)];
        color.a = uInBuffer.data[pos.y*uWidth*uChannel + pos.x*uChannel + (z+3)];

        imageStore(uImage, pos, color);
    }
}

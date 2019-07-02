layout(FORMAT, binding=0) readonly uniform PRECISION image3D uImage;

layout(binding=1) writeonly buffer destBuffer{
    float data[];
} uOutBuffer;

layout(location = 2) uniform int uWidth;
layout(location = 3) uniform int uHeight;
layout(location = 4) uniform int uChannel;

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (pos.x < uWidth && pos.y < uHeight)
    {
        vec4 color = imageLoad(uImage, pos);
        int z = pos.z*4;
        uOutBuffer.data[pos.y*uWidth*uChannel+pos.x*uChannel+(z+0)] = color.r;
        uOutBuffer.data[pos.y*uWidth*uChannel+pos.x*uChannel+(z+1)] = color.g;
        uOutBuffer.data[pos.y*uWidth*uChannel+pos.x*uChannel+(z+2)] = color.b;
        uOutBuffer.data[pos.y*uWidth*uChannel+pos.x*uChannel+(z+3)] = color.a;

    }
}

layout(FORMAT, binding=0) readonly uniform PRECISION image3D uImage;

layout(std430, binding=1) writeonly buffer destBuffer{
    vec4 data[];
} uOutBuffer;

layout(location = 2) uniform int uWidth;
layout(location = 3) uniform int uHeight;

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (pos.x < uWidth && pos.y < uHeight)
    {
        vec4 color = imageLoad(uImage, pos);
        uOutBuffer.data[uWidth*pos.y+pos.x+pos.z*uWidth*uHeight] = color;
    }
}

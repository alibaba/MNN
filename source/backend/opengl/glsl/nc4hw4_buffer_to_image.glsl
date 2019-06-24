layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uImage;

layout(binding=1) readonly buffer destBuffer{
    vec4 data[];
} uInBuffer;

layout(location = 2) uniform int uWidth;
layout(location = 3) uniform int uHeight;

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (pos.x < uWidth && pos.y < uHeight)
    {
        vec4 color = uInBuffer.data[uWidth*pos.y+pos.x+pos.z*uWidth*uHeight];
        imageStore(uImage, pos, color);
    }
}

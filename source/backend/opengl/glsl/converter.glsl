layout(std430) buffer;
layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(location=1) uniform mediump sampler3D uInput;
layout(location = 2) uniform int width;
layout(location = 3) uniform int height;
layout(location = 4) uniform int channel;

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (pos.x < width && pos.y < height && pos.z < channel)
    {
        vec4 result = texelFetch(uInput, pos, 0);
        imageStore(uOutput, pos, result);
    }
}

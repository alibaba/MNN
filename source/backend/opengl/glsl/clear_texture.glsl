layout(FORMAT, binding=0) writeonly uniform PRECISION image2D uOutput;
layout(location = 1) uniform int width;
layout(location = 2) uniform int height;

layout (local_size_x = 4, local_size_y = 4, local_size_z = 1) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (pos.x < width && pos.y < height)
    {
        imageStore(uOutput, ivec2(pos.x, pos.y), vec4(0,0,0,0));
    }
}

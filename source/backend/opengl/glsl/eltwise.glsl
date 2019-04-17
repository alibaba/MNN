
layout(FORMAT, binding=1) writeonly uniform PRECISION image3D uOutput;

layout(location=10) uniform ivec3 uOutputSize;
layout (local_size_x = 2, local_size_y = 2, local_size_z = 16) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 outputSize = uOutputSize;

    if (all(lessThan(pos, outputSize)))
    {
        vec4 color = MAINOP(pos);
        imageStore(uOutput, pos, color);
    }
}

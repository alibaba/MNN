layout(std430) buffer;
layout(FORMAT, binding=0) writeonly uniform PRECISION image2D uOutput;
layout(binding=2) readonly buffer kernel{
    vec4 data[];
} uKernel;

layout(location = 3) uniform int width;
layout(location = 4) uniform int height;

//index : ky * kx, oc/4, ic/4
//kernel buffer : oc ic h w -> oc/4 ic/4 ky kx ic4 oc4
//kernel image : oc/4, ky * kx * ic/4 * ic4
layout (local_size_x = 4, local_size_y = 4, local_size_z = 1) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (pos.x < width && pos.y < height)
    {
        vec4 res = uKernel.data[pos.x+pos.y*width];
        imageStore(uOutput, ivec2(pos.x, pos.y), res);
    }
}

layout(std430) buffer;
layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(binding=2) readonly buffer kernel{
    float data[];
} uKernel;

layout(location = 3) uniform int uFx;
layout(location = 4) uniform int uFy;

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    int fx = pos.y;
    int fy = pos.z;
    int z0 = pos.x * 4 + 0;
    int z1 = pos.x * 4 + 1;
    int z2 = pos.x * 4 + 2;
    int z3 = pos.x * 4 + 3;

    int p0 = z0*uFx*uFy + fy*uFx + fx;
    int p1 = z1*uFx*uFy + fy*uFx + fx;
    int p2 = z2*uFx*uFy + fy*uFx + fx;
    int p3 = z3*uFx*uFy + fy*uFx + fx;

    vec4 color = vec4(
        uKernel.data[p0],
        uKernel.data[p1],
        uKernel.data[p2],
        uKernel.data[p3]    
    );

    imageStore(uOutput, pos, color);
}

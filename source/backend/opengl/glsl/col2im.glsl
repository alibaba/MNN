layout(std430) buffer;
layout(binding=0, FORMAT) writeonly uniform mediump image3D uOutput;
layout(location=1) uniform mediump sampler2D uInput;
layout(binding=2) readonly buffer bias{
    vec4 data[];
} uBias;

layout(location=3) uniform ivec3 outputSize;

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;
#define UP_DIV(x, y) (((x)+(y)-1)/(y))

//index : ob*oc/4, oh, ow
//outputsize : oc/4, oh, ow
//input temp image : oc/4 * (ob*oh*ow)%4, (ob*oh*ow)/4 * oc4
void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    int ob = pos.z / outputSize.z;
    int oc_4 = pos.z % outputSize.z;

    if (all(lessThan(pos.xy, outputSize.xy)))
    {
        int sourceXIndex = ob*outputSize.x*outputSize.y + pos.y*outputSize.x + pos.x;
        int sourceX = sourceXIndex / 4;
        int sourceY = oc_4 * 4 + sourceXIndex % 4;

        vec4 color = uBias.data[pos.z];
        color += texelFetch(uInput, ivec2(sourceX, sourceY), 0);
#ifdef RELU
        color = max(color, vec4(0));
#endif
#ifdef RELU6
        color = clamp(color, vec4(0), vec4(6));
#endif
        imageStore(uOutput, pos, color);
    }
}

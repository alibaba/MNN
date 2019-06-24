layout(std430) buffer;
layout(binding=0, FORMAT) writeonly mediump uniform image2D uOutput;
layout(location=1) uniform mediump sampler3D uInput;
layout(location=5) uniform int ic_4;
layout(location=6) uniform int outputWidth;
layout(location=7) uniform int outputHeight;

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

//index : ib*ic/4, oh, ow
//input image ic/4, ih, iw * ic4
//output : temp image : (ib*oh*ow)/ 4, ic/4*(ib*oh*ow)%4*ic4
void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    if (pos.x < outputWidth && pos.y < outputHeight)
    {
        int ic_4_i = pos.z % ic_4;
        int ib_i = pos.z / ic_4;
        int destYOrigin = ib_i*outputWidth*outputHeight + pos.y*outputWidth + pos.x;
        int destY = destYOrigin / 4;
        int destXOffset = destYOrigin % 4;
        vec4 color = texelFetch(uInput, ivec3(pos.x, pos.y, pos.z), 0);
        imageStore(uOutput, ivec2(ic_4_i*4+destXOffset, destY), color);
    }
}

layout(std430) buffer;
layout(binding=0, FORMAT) writeonly mediump uniform image2D uOutput;
layout(location=1) uniform mediump sampler3D uInput;
layout(location=2) uniform ivec2 pad;
layout(location=3) uniform ivec2 kernelSize;
layout(location=4) uniform ivec2 stride;
layout(location=5) uniform ivec2 dilate;
layout(location=6) uniform ivec4 inputSize;
layout(location=7) uniform ivec4 outputSize;

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

//index : ib*ic/4, oh, ow
//input image ic/4, ih, iw * ic4
//inputsize : ic/4, ih, iw
//outputsize : oc/4, oh, ow
//output : temp image : (ib*oh*ow)/ 4, ic/4*ky*kx*(ib*oh*ow)%4*ic4
void main()
{
    ivec3 index = ivec3(gl_GlobalInvocationID);
    if (index.x < outputSize.x && index.y < outputSize.y)
    {
        ivec2 s0 = index.xy*stride-pad;
        ivec2 sfxy = max(ivec2(0), (UP_DIV(-s0, dilate)));
        ivec2 efxy = min(kernelSize, UP_DIV(inputSize.xy-s0, dilate));
        int ic_4 = index.z % inputSize.z; //input channel
        int ib = index.z / inputSize.z; // input batch
        
        int destYOrigin = ib*outputSize.x*outputSize.y + index.y*outputSize.x + index.x;
        int destY = destYOrigin / 4;
        int destXOffset = destYOrigin % 4;
        for (int fy=0; fy<kernelSize.y; ++fy)
        {
            int sy = fy*dilate.y + s0.y;
            for (int fx=0; fx<kernelSize.x; ++fx)
            {
                int sx = fx*dilate.x + s0.x;
                int destX = fx + fy*kernelSize.x + ic_4*kernelSize.x * kernelSize.y;
                vec4 color = texelFetch(uInput, ivec3(sx, sy, index.z), 0);
                imageStore(uOutput, ivec2(4*destX+destXOffset, destY), color);
            }
        }
    }
}

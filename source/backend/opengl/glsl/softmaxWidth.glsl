
layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(location=1) uniform mediump sampler3D uInput;
layout(location=2) uniform int w;
layout(location=3) uniform int h;
layout(location=4) uniform int c;

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

void main()
{
    // input tensor's layout is NC4HW4
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    
    int channelDiv4 = (c + 3) / 4;
    int HW = w * h;
    
    if(pos.y < h && pos.z < channelDiv4)
    {
        // get the max value
        vec4 maxValue = vec4(-1000.0);
        for(int i = 0; i < w; ++i)
        {
            maxValue = max(maxValue, texelFetch(uInput, ivec3(i, pos.y, pos.z), 0));
        }
        
        // sum
        vec4 sum = vec4(0.0);
        for(int i = 0; i < w; ++i)
        {
            sum += exp(texelFetch(uInput, ivec3(i, pos.y, pos.z), 0) - maxValue);
        }
        // div
        for(int i = 0; i < w; ++i)
        {
            ivec3 curPos = ivec3(i, pos.y, pos.z);
            imageStore(uOutput, curPos, exp(texelFetch(uInput, ivec3(i, pos.y, pos.z), 0) - maxValue) / sum);
        }
        
    }
}

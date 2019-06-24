
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
    
    if(pos.x < w && pos.y < h)
    {
        int channelDiv4 = c / 4;
        int upDiv4 = (c + 3) / 4;
        int lastChannel = c % 4;
        int batchIndex = pos.z * upDiv4;
        // get the max value
        vec4 maxValue = vec4(-1000.0);
        for(int i = 0; i < channelDiv4; ++i)
        {
            maxValue = max(maxValue, texelFetch(uInput, ivec3(pos.x, pos.y, i + batchIndex), 0));
        }
        // get the true max vaule
        float maxValueTrue = -1000.0;
        
        maxValueTrue = max(maxValue.x, maxValue.y);
        maxValueTrue = max(maxValueTrue, maxValue.z);
        maxValueTrue = max(maxValueTrue, maxValue.w);
        
        if(lastChannel == 1)
        {
            vec4 tempData = texelFetch(uInput, ivec3(pos.x, pos.y, channelDiv4 + batchIndex), 0);
            maxValueTrue = max(maxValueTrue, tempData.x);
        }
        else if(lastChannel == 2)
        {
            vec4 tempData = texelFetch(uInput, ivec3(pos.x, pos.y, channelDiv4 + batchIndex), 0);
            maxValueTrue = max(maxValueTrue, tempData.x);
            maxValueTrue = max(maxValueTrue, tempData.y);
        }
        else
        {
            vec4 tempData = texelFetch(uInput, ivec3(pos.x, pos.y, channelDiv4 + batchIndex), 0);
            maxValueTrue = max(maxValueTrue, tempData.x);
            maxValueTrue = max(maxValueTrue, tempData.y);
            maxValueTrue = max(maxValueTrue, tempData.z);
        }
        
        // exp
        maxValue = vec4(maxValueTrue);
        vec4 sum = vec4(0.0);
        for(int i = 0; i < channelDiv4; ++i)
        {
            sum += exp(texelFetch(uInput, ivec3(pos.x, pos.y, i + batchIndex), 0) - maxValue);
        }
        
        float sumTrue = 0.0;
        sumTrue = sum.x + sum.y + sum.z + sum.w;
        
        if(lastChannel == 1)
        {
            vec4 tempData = texelFetch(uInput, ivec3(pos.x, pos.y, channelDiv4 + batchIndex), 0);
            sumTrue += exp(tempData.x - maxValueTrue);
        }
        else if(lastChannel == 2)
        {
            vec4 tempData = texelFetch(uInput, ivec3(pos.x, pos.y, channelDiv4 + batchIndex), 0);
            sumTrue += (exp(tempData.x - maxValueTrue) + exp(tempData.y - maxValueTrue));
        }
        else
        {
            vec4 tempData = texelFetch(uInput, ivec3(pos.x, pos.y, channelDiv4 + batchIndex), 0);
            sumTrue += (exp(tempData.x - maxValueTrue) + exp(tempData.y - maxValueTrue) + exp(tempData.z - maxValueTrue));
        }
        
        // div sum
        sum = vec4(sumTrue);
        for(int i = 0; i < upDiv4; ++i)
        {
            ivec3 curPos = ivec3(pos.x, pos.y, i + batchIndex);
            imageStore(uOutput, curPos, exp(texelFetch(uInput, curPos, 0) - maxValue) / sum);
        }
        
    }
}

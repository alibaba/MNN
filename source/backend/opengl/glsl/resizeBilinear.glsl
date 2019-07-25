
layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(location=1) uniform mediump sampler3D uInput;
layout(location=2) uniform ivec4 inImgSize;
layout(location=3) uniform ivec4 outImgSize;
layout(location=4) uniform vec2 scale;

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 inputImgSize = inImgSize.xyz;
    ivec3 outputImgSize = outImgSize.xyz;
    
    if(pos.x < outputImgSize.x && pos.y < outputImgSize.y && pos.z < outputImgSize.z)
    {
        float srcX = float(pos.x) * scale.x;
        int x1 = int(floor(srcX));
        int x11 = clamp(x1, 0, inputImgSize.x - 1);
        int x12 = clamp(x1 + 1, 0, inputImgSize.x - 1);
        vec4 factorX = vec4(srcX - float(x1));

        float srcY = float(pos.y) * scale.y;
        int y1 = int(floor(srcY));
        int y11 = clamp(y1, 0, inputImgSize.y - 1);
        int y12 = clamp(y1 + 1, 0, inputImgSize.y - 1);
        vec4 factorY = vec4(srcY - float(y1));

        vec4 res1 = texelFetch(uInput, ivec3(x11, y12, pos.z), 0);
        vec4 res2 = texelFetch(uInput, ivec3(x12, y12, pos.z), 0);
        vec4 res3 = texelFetch(uInput, ivec3(x11, y11, pos.z), 0);
        vec4 res4 = texelFetch(uInput, ivec3(x12, y11, pos.z), 0);
        vec4 res11 = (vec4(1.0) - factorX) * res1 + factorX * res2;
        vec4 res12 = (vec4(1.0) - factorX) * res3 + factorX * res4;
        
        imageStore(uOutput, pos, factorY * res11 + (vec4(1.0) - factorY) * res12);
    }
    
}

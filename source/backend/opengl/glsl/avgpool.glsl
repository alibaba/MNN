layout(FORMAT, binding=0, location=0) readonly uniform PRECISION image3D uInput;
layout(FORMAT, binding=1, location=1) writeonly uniform PRECISION image3D uOutput;

layout(location = 2) uniform ivec2 uKernel;
layout(location = 3) uniform ivec2 uStride;
layout(location = 4) uniform ivec2 uPad;
layout(location=10) uniform ivec3 uOutputSize;
layout(location=11) uniform ivec3 uInputSize;

layout (local_size_x = 2, local_size_y = 2, local_size_z = 16) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 outputSize = uOutputSize;
    ivec2 spos = pos.xy*uStride-uPad;

    if (all(lessThan(pos, outputSize)))
    {
        ivec2 inputSizeXY = uInputSize.xy;
        vec4 color = vec4(0.0);
        vec4 num = vec4(0.0);
        ivec2 sfxy = max(ivec2(0), -spos);
        ivec2 efxy = min(uKernel, inputSizeXY-spos);

        for (int fy=sfxy.y; fy<efxy.y; ++fy)
        {
            for (int fx=sfxy.x; fx<efxy.x; ++fx)
            {
                ivec2 spos_ = spos + ivec2(fx, fy);
                color += imageLoad(uInput, ivec3(spos.x+fx, spos.y+fy, pos.z));
                num += vec4(1.0);
            }
        }
        imageStore(uOutput, pos, color/num);
    }
}

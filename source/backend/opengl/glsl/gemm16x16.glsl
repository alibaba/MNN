layout(std430) buffer;
layout(binding=0, FORMAT) writeonly mediump uniform image2D uOutput;
layout(binding=1, FORMAT) readonly mediump uniform image2D uInput;
layout(binding=2, FORMAT) readonly mediump uniform image2D uKernel;

layout(location=3) uniform ivec2 outputSize;
layout(location=4) uniform int ic_4;

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

//index : 1, oc/4, (ob*oh*ow)/4
//outputsize :  oc/4, (ob*oh*ow)/4
//multiLength : ci/4
//kernel image : oc/4, ic/4 * ic4  * oc4
//input : temp image : (ib*oh*ow)/ 4, ic/4*(ib*oh*ow)%4*ic4
//output : temp image : oc/4 * (ob*oh*ow)%4, (ob*oh*ow)/4 * oc4
void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID); // 1, oc/4, (ob*oh*ow)/4
    int oc_4 = pos.y;
    int obxohxow_4 = pos.x;
    if (obxohxow_4 < outputSize.x && oc_4 < outputSize.y)
    {
        vec4 o0 = vec4(0);
        vec4 o1 = vec4(0);
        vec4 o2 = vec4(0);
        vec4 o3 = vec4(0);

        for (int k=0; k<ic_4; ++k)
        {
            int k4 = k << 2;
            vec4 k0 = imageLoad(uKernel, ivec2(k4, oc_4));
            vec4 s0 = imageLoad(uInput, ivec2(k4++, obxohxow_4));
            vec4 k1 = imageLoad(uKernel, ivec2(k4, oc_4));
            vec4 s1 = imageLoad(uInput, ivec2(k4++, obxohxow_4));
            vec4 k2 = imageLoad(uKernel, ivec2(k4, oc_4));
            vec4 s2 = imageLoad(uInput, ivec2(k4++, obxohxow_4));
            vec4 k3 = imageLoad(uKernel, ivec2(k4, oc_4));
            vec4 s3 = imageLoad(uInput, ivec2(k4, obxohxow_4));

            mat4 kernel_mat = mat4(k0, k1, k2, k3);

            o0 += kernel_mat * s0;
            o1 += kernel_mat * s1;
            o2 += kernel_mat * s2;
            o3 += kernel_mat * s3;
        }
        int oc_44 = oc_4 << 2;
        imageStore(uOutput, ivec2(obxohxow_4, oc_44++), o0);
        imageStore(uOutput, ivec2(obxohxow_4, oc_44++), o1);
        imageStore(uOutput, ivec2(obxohxow_4, oc_44++), o2);
        imageStore(uOutput, ivec2(obxohxow_4, oc_44++), o3);
        
    }
}

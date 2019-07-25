layout(location=0) uniform mediump sampler3D uInput;
layout(FORMAT, binding=1) writeonly uniform PRECISION image3D uOutput;
layout(location=2) uniform mediump sampler3D uRoI;

layout(location=10) uniform ivec3 uOutputSize;
layout(location=11) uniform ivec3 uInputSize;
layout(location=12) uniform float spatialScale;

layout (local_size_x = XLOCAL, local_size_y = YLOCAL, local_size_z = ZLOCAL) in;

void main()
{
    ivec3 pos = ivec3(gl_GlobalInvocationID);

    if(pos.x < uOutputSize.x && pos.y < uOutputSize.y)
    {
        ivec3 uInputSize = uInputSize.xyz;
        int roiBatchIndex = pos.z / uInputSize.z;
        int inputZIndex = pos.z % uInputSize.z;
        // 0, xmin, ymin, xmax, ymax
        vec4 roiData0 = texelFetch(uRoI, ivec3(0, 0, roiBatchIndex), 0);
        vec4 roiData1 = texelFetch(uRoI, ivec3(0, 0, roiBatchIndex + 1), 0);
        int x1 = int(round(float(roiData0.y) * spatialScale));
        int y1 = int(round(float(roiData0.z) * spatialScale));
        int x2 = int(round(float(roiData0.w) * spatialScale));
        int y2 = int(round(float(roiData1.x) * spatialScale));

        int roiW = max(x2 - x1 + 1, 1);
        int roiH = max(y2 - y1 + 1, 1);
        float binSizeW = float(roiW) / float(uOutputSize.x);
        float binSizeH = float(roiH) / float(uOutputSize.y);

        int wStart = clamp(x1 + int(floor(float(pos.x) * binSizeW)), 0, uInputSize.x);
        int wEnd = clamp(x1 + int(ceil(float(pos.x + 1) * binSizeW)), 0, uInputSize.x);
        int hStart = clamp(y1 + int(floor(float(pos.y) * binSizeH)), 0, uInputSize.y);
        int hEnd = clamp(y1 + int(ceil(float(pos.y + 1) * binSizeH)), 0, uInputSize.y);

        bool isEmpty = (hEnd <= hStart) || (wEnd <= wStart);
        vec4 res = isEmpty ? vec4(0.0) : texelFetch(uInput, ivec3(0, 0, inputZIndex), 0);

        for(int i = hStart; i < hEnd; ++i)
        {
            for(int j = wStart; j < wEnd; ++j)
            {
                res = max(res, texelFetch(uInput, ivec3(j, i, inputZIndex), 0));
            }
        }

        imageStore(uOutput, pos, res);
    }

}

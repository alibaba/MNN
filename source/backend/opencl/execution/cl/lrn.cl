__kernel void lrn_buffer(__global float *inputTempPtr,
                        __global float *outputTempPtr,
                        __private const int4 imgSize,
                        __private const int localSize,
                        __private const float alpha,
                        __private const float beta)
{
	int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	int3 imgInfo = imgSize.xyz;
    int hw = imgInfo.x*imgInfo.y;
	
	if(pos.x < imgInfo.x && pos.y < imgInfo.y)
	{
		float sum = 0.0f;
		int halfSize = localSize/2;
		for(int c = pos.z - halfSize; c < pos.z + halfSize; c++)
		{
			if(c < 0 || c >= imgInfo.z) continue;
			int index = pos.x + pos.y * imgInfo.x + c * hw;
			sum += inputTempPtr[index] * inputTempPtr[index];
		}

		int dataIndex = pos.x + pos.y * imgInfo.x + pos.z * hw;
		outputTempPtr[dataIndex] = inputTempPtr[dataIndex] * pow(1.0f + alpha * sum, -beta);
	}
}

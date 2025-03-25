#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void batch_gather_buf(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                            __global OUTPUT_TYPE* output, __global INPUT_TYPE* input,
                            #ifdef OFFSET_DST
                            __global int* offset_dst_ptr,
                            __private const int4 offset_dst_shape,// w, h, c, n
                            #endif
                            #ifdef OFFSET_SRC
                            __global int* offset_src_ptr,
                            __private const int4 offset_src_shape,// w, h, c, n
                            #endif
                            __private const int x_size,
                            __private const int4 stride_src,
                            __private const int4 stride_dst,
                            __private const int2 steps,
                            __private const int2 iters,
                            __private const int inputSize) {
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        
        int x = pos.x % x_size;
        int y = pos.x / x_size;

        int2 index = (int2)(pos.z, pos.z);
#ifdef OFFSET_DST
        index.x = offset_dst_ptr[pos.z];
#endif
            
#ifdef OFFSET_SRC
        index.y = offset_src_ptr[pos.z];
#endif
        int2 offset = index * steps;
        int src_offset = offset.y + stride_src.w + x * stride_src.x + y * stride_src.y + pos.y * stride_src.z;
        int dst_offset = offset.x + stride_dst.w + x * stride_dst.x + y * stride_dst.y + pos.y * stride_dst.z;

        if(offset.x >= 0){
            if(offset.y >= 0 && offset.y < inputSize){
                output[dst_offset] = (OUTPUT_TYPE)input[src_offset];
            }else{
                output[dst_offset] = (OUTPUT_TYPE)(0);
            }
        }
    }
}

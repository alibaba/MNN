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
                            __private const int4 dst_c4size,// w, h, c, n
                            __private const int4 src_c4size,// w, h, c, n
                            __private const int inputSize) {
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        
        int x = pos.x % x_size;
        int y = pos.x / x_size;

        int2 index = (int2)(pos.z, pos.z);
        
        #ifdef OFFSET_DST
        {
            int offset_value = pos.z;
            int off_c4_size = (offset_dst_shape.z + 3) >> 2;
            #ifdef GATHER_INPUT_NHWC
            int off_c = offset_value % offset_dst_shape.z; offset_value /= offset_dst_shape.z;
            int off_w = offset_value % offset_dst_shape.x; offset_value /= offset_dst_shape.x;
            int off_h = offset_value % offset_dst_shape.y;
            int off_b = offset_value / offset_dst_shape.y;
            #else
            int off_w = offset_value % offset_dst_shape.x; offset_value /= offset_dst_shape.x;
            int off_h = offset_value % offset_dst_shape.y; offset_value /= offset_dst_shape.y;
            int off_c = offset_value % offset_dst_shape.z;
            int off_b = offset_value / offset_dst_shape.z;
            #endif
            int real_dst_offset = (((off_b * off_c4_size + off_c / 4) * offset_dst_shape.y + off_h) * offset_dst_shape.x + off_w) * 4 + off_c % 4;
            index.x = offset_dst_ptr[real_dst_offset];
        }
        #endif
    
        #ifdef OFFSET_SRC
        {
            int offset_value = pos.z;
            int off_c4_size = (offset_src_shape.z + 3) >> 2;
            #ifdef GATHER_INPUT_NHWC
            int off_c = offset_value % offset_src_shape.z; offset_value /= offset_src_shape.z;
            int off_w = offset_value % offset_src_shape.x; offset_value /= offset_src_shape.x;
            int off_h = offset_value % offset_src_shape.y;
            int off_b = offset_value / offset_src_shape.y;
            #else
            int off_w = offset_value % offset_src_shape.x; offset_value /= offset_src_shape.x;
            int off_h = offset_value % offset_src_shape.y; offset_value /= offset_src_shape.y;
            int off_c = offset_value % offset_src_shape.z;
            int off_b = offset_value / offset_src_shape.z;
            #endif
            int real_src_offset = (((off_b * off_c4_size + off_c / 4) * offset_src_shape.y + off_h) * offset_src_shape.x + off_w) * 4 + off_c % 4;
            index.y = offset_src_ptr[real_src_offset];
        }
        #endif
    
        int2 offset = index * steps;
        int src_offset = offset.y + stride_src.w + x * stride_src.x + y * stride_src.y + pos.y * stride_src.z;
        int dst_offset = offset.x + stride_dst.w + x * stride_dst.x + y * stride_dst.y + pos.y * stride_dst.z;

        int src_offsetC4, dst_offsetC4;
        {
#ifdef GATHER_INPUT_NHWC
            int c = src_offset % src_c4size.z; src_offset /= src_c4size.z;
            int w = src_offset % src_c4size.x; src_offset /= src_c4size.x;
            int h = src_offset % src_c4size.y;
            int b = src_offset / src_c4size.y;
            int c4_size = (src_c4size.z + 3) / 4;
            src_offsetC4 = (((b * c4_size + (c / 4)) * src_c4size.y + h) * src_c4size.x + w) * 4 + (c % 4);
#else
            int w = src_offset % src_c4size.x; src_offset /= src_c4size.x;
            int h = src_offset % src_c4size.y; src_offset /= src_c4size.y;
            int c = src_offset % src_c4size.z;
            int b = src_offset / src_c4size.z;
            int c4_size = (src_c4size.z + 3) / 4;
            src_offsetC4 = (((b * c4_size + (c / 4)) * src_c4size.y + h) * src_c4size.x + w) * 4 + (c % 4);
#endif
        }
        {
#ifdef GATHER_OUTPUT_NHWC
            int c = dst_offset % dst_c4size.z; dst_offset /= dst_c4size.z;
            int w = dst_offset % dst_c4size.x; dst_offset /= dst_c4size.x;
            int h = dst_offset % dst_c4size.y;
            int b = dst_offset / dst_c4size.y;
            int c4_size = (dst_c4size.z + 3) / 4;
            dst_offsetC4 = (((b * c4_size + (c / 4)) * dst_c4size.y + h) * dst_c4size.x + w) * 4 + (c % 4);
#else
            int w = dst_offset % dst_c4size.x; dst_offset /= dst_c4size.x;
            int h = dst_offset % dst_c4size.y; dst_offset /= dst_c4size.y;
            int c = dst_offset % dst_c4size.z;
            int b = dst_offset / dst_c4size.z;
            int c4_size = (dst_c4size.z + 3) / 4;
            dst_offsetC4 = (((b * c4_size + (c / 4)) * dst_c4size.y + h) * dst_c4size.x + w) * 4 + (c % 4);
#endif
        }
        if(offset.x >= 0){
            if(offset.y >= 0 && offset.y < inputSize){
                output[dst_offsetC4] = (OUTPUT_TYPE)input[src_offsetC4];
            }else{
                output[dst_offsetC4] = (OUTPUT_TYPE)(0);
            }
        }
    }
}

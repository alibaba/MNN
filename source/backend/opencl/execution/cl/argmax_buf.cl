#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define ARGMAX_SELECT(A, B, C, D)          \
    if(A.x < B.x){ A.x = B.x; C.x = D; }    \
    if(A.y < B.y){ A.y = B.y; C.y = D; }    \
    if(A.z < B.z){ A.z = B.z; C.z = D; }    \
    if(A.w < B.w){ A.w = B.w; C.w = D; }    

#define ARGMIN_SELECT(A, B, C, D)    \
    if(A.x > B.x){ A.x = B.x; C.x = D; }    \
    if(A.y > B.y){ A.y = B.y; C.y = D; }    \
    if(A.z > B.z){ A.z = B.z; C.z = D; }    \
    if(A.w > B.w){ A.w = B.w; C.w = D; }    


__kernel void argmax_buf(GLOBAL_SIZE_3_DIMS
                        __global const FLOAT* input,
                        __global int* output,
                        __private const int inside,
                        __private const int outside,
                        __private const int dim){
    const int x = get_global_id(0);
    const int y = get_global_id(1); // inside
    const int z = get_global_id(2); // outside
    
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    int index = 0;
#ifdef ARGMAX
    FLOAT maxValue = (FLOAT)-FLT_MAX;
#else
FLOAT maxValue = (FLOAT)FLT_MAX;
#endif
    const int offset = z * dim * inside + y;
#if ARGMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    FLOAT local reduce[ARGMAX_LOCAL_SIZE];
    int local index_reduce[ARGMAX_LOCAL_SIZE];
        
    for (int i=lid; i < dim; i+=ARGMAX_LOCAL_SIZE) {
        FLOAT value = input[offset + i * inside];
#ifdef ARGMAX
        if(maxValue < value){ maxValue = value; index = i; }
#else
        if(maxValue > value){ maxValue = value; index = i; }
#endif
    }
    reduce[lid] = maxValue;
    index_reduce[lid] = index;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = ARGMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i){
#ifdef ARGMAX
            if(reduce[lid] < reduce[lid + i]){reduce[lid] = reduce[lid + i]; index_reduce[lid] = index_reduce[lid + i];}
#else
            if(reduce[lid] > reduce[lid + i]){reduce[lid] = reduce[lid + i]; index_reduce[lid] = index_reduce[lid + i];}
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0){
        output[z * inside + y] = index_reduce[0];
    }
#else
    for(int i = 0; i < dim; ++i){
        FLOAT value = input[ + offset + i * inside];
#ifdef ARGMAX
        if(maxValue < value){ maxValue = value; index = i; }
#else
        if(maxValue > value){ maxValue = value; index = i; }
#endif
    }
    output[z * inside + y] = index;
#endif
}


__kernel void argmax_v4_buf(GLOBAL_SIZE_3_DIMS
                        __global const FLOAT* input,
                        __global int* output,
                        __private const int inside,
                        __private const int outside,
                        __private const int dim){
    const int x = get_global_id(0);
    const int y = get_global_id(1) << 2; // inside
    const int z = get_global_id(2); // outside
    
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    int4 index = 0;
#ifdef ARGMAX
    FLOAT4 maxValue = (FLOAT4)-FLT_MAX;
#else
    FLOAT4 maxValue = (FLOAT4)FLT_MAX;
#endif
    const int offset = z * dim * inside + y;
#if ARGMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    FLOAT4 local reduce[ARGMAX_LOCAL_SIZE];
    int4 local index_reduce[ARGMAX_LOCAL_SIZE];
        
    for (int i=lid; i < dim; i+=ARGMAX_LOCAL_SIZE) {
        FLOAT4 value = vload4(0, input + offset + i * inside);
#ifdef ARGMAX
        ARGMAX_SELECT(maxValue, value, index, i);
#else
        ARGMIN_SELECT(maxValue, value, index, i);
#endif
    }
    reduce[lid] = maxValue;
    index_reduce[lid] = index;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = ARGMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i){
#ifdef ARGMAX
            if(reduce[lid].x < reduce[lid + i].x){reduce[lid].x = reduce[lid + i].x; index_reduce[lid].x = index_reduce[lid + i].x;}
            if(reduce[lid].y < reduce[lid + i].y){reduce[lid].y = reduce[lid + i].y; index_reduce[lid].y = index_reduce[lid + i].y;}
            if(reduce[lid].z < reduce[lid + i].z){reduce[lid].z = reduce[lid + i].z; index_reduce[lid].z = index_reduce[lid + i].z;}
            if(reduce[lid].w < reduce[lid + i].w){reduce[lid].w = reduce[lid + i].w; index_reduce[lid].w = index_reduce[lid + i].w;}
#else
            if(reduce[lid].x > reduce[lid + i].x){reduce[lid].x = reduce[lid + i].x; index_reduce[lid].x = index_reduce[lid + i].x;}
            if(reduce[lid].y > reduce[lid + i].y){reduce[lid].y = reduce[lid + i].y; index_reduce[lid].y = index_reduce[lid + i].y;}
            if(reduce[lid].z > reduce[lid + i].z){reduce[lid].z = reduce[lid + i].z; index_reduce[lid].z = index_reduce[lid + i].z;}
            if(reduce[lid].w > reduce[lid + i].w){reduce[lid].w = reduce[lid + i].w; index_reduce[lid].w = index_reduce[lid + i].w;}
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0){
        vstore4(index_reduce[0], 0, output + z * inside + y);
    }
#else
    for(int i = 0; i < dim; ++i){
        FLOAT4 value = vload4(0, input + offset + i * inside);
#ifdef ARGMAX
        ARGMAX_SELECT(maxValue, value, index, i);
#else
        ARGMIN_SELECT(maxValue, value, index, i);
#endif
    }
    vstore4(index, 0, output + z * inside + y);
#endif
}

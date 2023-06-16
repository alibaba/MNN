/* Copyright (c) 2009-2022 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */

#ifndef __OPENCL_CL_EXT_QCOM_H
#define __OPENCL_CL_EXT_QCOM_H

#include <CL/cl_ext.h>

#ifdef __cplusplus
extern "C" {
#endif


/************************************
 * cl_qcom_create_buffer_from_image *
 ************************************/

#define CL_BUFFER_FROM_IMAGE_ROW_PITCH_QCOM         0x40C0
#define CL_BUFFER_FROM_IMAGE_SLICE_PITCH_QCOM       0x40C1

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateBufferFromImageQCOM(cl_mem       image,
                            cl_mem_flags flags,
                            cl_int      *errcode_ret);


/************************************
 * cl_qcom_limited_printf extension *
 ************************************/

/* Builtin printf function buffer size in bytes. */
#define CL_DEVICE_PRINTF_BUFFER_SIZE_QCOM           0x1049


/*************************************
 * cl_qcom_extended_images extension *
 *************************************/

#define CL_CONTEXT_ENABLE_EXTENDED_IMAGES_QCOM      0x40AA
#define CL_DEVICE_EXTENDED_IMAGE2D_MAX_WIDTH_QCOM   0x40AB
#define CL_DEVICE_EXTENDED_IMAGE2D_MAX_HEIGHT_QCOM  0x40AC
#define CL_DEVICE_EXTENDED_IMAGE3D_MAX_WIDTH_QCOM   0x40AD
#define CL_DEVICE_EXTENDED_IMAGE3D_MAX_HEIGHT_QCOM  0x40AE
#define CL_DEVICE_EXTENDED_IMAGE3D_MAX_DEPTH_QCOM   0x40AF

/*************************************
 * cl_qcom_perf_hint extension *
 *************************************/

typedef cl_uint                                     cl_perf_hint;

#define CL_CONTEXT_PERF_HINT_QCOM                   0x40C2

/*cl_perf_hint*/
#define CL_PERF_HINT_HIGH_QCOM                      0x40C3
#define CL_PERF_HINT_NORMAL_QCOM                    0x40C4
#define CL_PERF_HINT_LOW_QCOM                       0x40C5

extern CL_API_ENTRY cl_int CL_API_CALL
clSetPerfHintQCOM(cl_context    context,
                  cl_perf_hint  perf_hint);

// This extension is published at Khronos, so its definitions are made in cl_ext.h.
// This duplication is for backward compatibility.

#ifndef CL_MEM_ANDROID_NATIVE_BUFFER_HOST_PTR_QCOM

/*********************************
* cl_qcom_android_native_buffer_host_ptr extension
*********************************/

#define CL_MEM_ANDROID_NATIVE_BUFFER_HOST_PTR_QCOM                  0x40C6


typedef struct _cl_mem_android_native_buffer_host_ptr
{
    // Type of external memory allocation.
    // Must be CL_MEM_ANDROID_NATIVE_BUFFER_HOST_PTR_QCOM for Android native buffers.
    cl_mem_ext_host_ptr  ext_host_ptr;

    // Virtual pointer to the android native buffer
    void*                anb_ptr;

} cl_mem_android_native_buffer_host_ptr;

#endif   //#ifndef CL_MEM_ANDROID_NATIVE_BUFFER_HOST_PTR_QCOM

#define CL_MEM_PMEM_HOST_PTR_QCOM                  0x4116

typedef struct _cl_mem_pmem_host_ptr
{
    /* Type of external memory allocation. */
    /* Must be CL_MEM_PMEM_HOST_PTR_QCOM for PMEM allocations. */
    cl_mem_ext_host_ptr  ext_host_ptr;

    /* PMEM handle */
    uintptr_t            pmem_handle;

    /* Host pointer to the PMEM allocated memory */
    void*                pmem_hostptr;

} cl_mem_pmem_host_ptr;

/*********************************
* cl_qcom_other_image extension
*********************************/

// Extended flag for creating/querying QCOM non-standard images
#define CL_MEM_OTHER_IMAGE_QCOM                             (1ULL << 37)

// cl_channel_type
#define CL_QCOM_UNORM_MIPI10                                0x4159
#define CL_QCOM_UNORM_MIPI12                                0x415A
#define CL_QCOM_UNSIGNED_MIPI10                             0x415B
#define CL_QCOM_UNSIGNED_MIPI12                             0x415C
#define CL_QCOM_UNORM_INT10                                 0x415D
#define CL_QCOM_UNORM_INT12                                 0x415E
#define CL_QCOM_UNSIGNED_INT16                              0x415F

// cl_channel_order
// Dedicate 0x4130-0x415F range for QCOM extended image formats
// 0x4130 - 0x4132 range is assigned to pixel-oriented compressed format
#define CL_QCOM_BAYER                                       0x414E

#define CL_QCOM_NV12                                        0x4133
#define CL_QCOM_NV12_Y                                      0x4134
#define CL_QCOM_NV12_UV                                     0x4135

#define CL_QCOM_TILED_NV12                                  0x4136
#define CL_QCOM_TILED_NV12_Y                                0x4137
#define CL_QCOM_TILED_NV12_UV                               0x4138

#define CL_QCOM_P010                                        0x413C
#define CL_QCOM_P010_Y                                      0x413D
#define CL_QCOM_P010_UV                                     0x413E

#define CL_QCOM_TILED_P010                                  0x413F
#define CL_QCOM_TILED_P010_Y                                0x4140
#define CL_QCOM_TILED_P010_UV                               0x4141


#define CL_QCOM_TP10                                        0x4145
#define CL_QCOM_TP10_Y                                      0x4146
#define CL_QCOM_TP10_UV                                     0x4147

#define CL_QCOM_TILED_TP10                                  0x4148
#define CL_QCOM_TILED_TP10_Y                                0x4149
#define CL_QCOM_TILED_TP10_UV                               0x414A

#define CL_QCOM_NV12_512                                    0x4152
#define CL_QCOM_NV12_512_Y                                  0x4153
#define CL_QCOM_NV12_512_UV                                 0x4154

/*********************************
* cl_qcom_compressed_image extension
*********************************/

// Extended flag for creating/querying QCOM non-planar compressed images
#define CL_MEM_COMPRESSED_IMAGE_QCOM                        (1ULL << 38)

// Extended image format
// cl_channel_order
#define CL_QCOM_COMPRESSED_RGBA                             0x4130
#define CL_QCOM_COMPRESSED_RGBx                             0x4131

#define CL_QCOM_COMPRESSED_NV12_Y                           0x413A
#define CL_QCOM_COMPRESSED_NV12_UV                          0x413B

#define CL_QCOM_COMPRESSED_P010                             0x4142
#define CL_QCOM_COMPRESSED_P010_Y                           0x4143
#define CL_QCOM_COMPRESSED_P010_UV                          0x4144

#define CL_QCOM_COMPRESSED_TP10                             0x414B
#define CL_QCOM_COMPRESSED_TP10_Y                           0x414C
#define CL_QCOM_COMPRESSED_TP10_UV                          0x414D

#define CL_QCOM_COMPRESSED_NV12_4R                          0x414F
#define CL_QCOM_COMPRESSED_NV12_4R_Y                        0x4150
#define CL_QCOM_COMPRESSED_NV12_4R_UV                       0x4151
/*********************************
* cl_qcom_compressed_yuv_image_read extension
*********************************/

// Extended flag for creating/querying QCOM compressed images
#define CL_MEM_COMPRESSED_YUV_IMAGE_QCOM                    (1ULL << 39)

// Extended image format
#define CL_QCOM_COMPRESSED_NV12                             0x4139

// Extended flag for setting ION buffer allocation type
#define CL_MEM_ION_HOST_PTR_COMPRESSED_YUV_QCOM                 0x40CD
#define CL_MEM_ION_HOST_PTR_PROTECTED_COMPRESSED_YUV_QCOM       0x40CE

/*********************************
* cl_qcom_accelerated_image_ops
*********************************/
#define CL_MEM_OBJECT_WEIGHT_IMAGE_QCOM                         0x4110
#define CL_DEVICE_HOF_MAX_NUM_PHASES_QCOM                       0x4111
#define CL_DEVICE_HOF_MAX_FILTER_SIZE_X_QCOM                    0x4112
#define CL_DEVICE_HOF_MAX_FILTER_SIZE_Y_QCOM                    0x4113
#define CL_DEVICE_BLOCK_MATCHING_MAX_REGION_SIZE_X_QCOM         0x4114
#define CL_DEVICE_BLOCK_MATCHING_MAX_REGION_SIZE_Y_QCOM         0x4115

//Extended flag for specifying weight image type
#define CL_WEIGHT_IMAGE_SEPARABLE_QCOM                          (1<<0)

// Box Filter
typedef struct _cl_box_filter_size_qcom
{
    // Width of box filter on X direction.
    float box_filter_width;

    // Height of box filter on Y direction.
    float box_filter_height;
} cl_box_filter_size_qcom;

// HOF Weight Image Desc
typedef struct _cl_weight_desc_qcom
{
    /** Coordinate of the "center" point of the weight image,
        based on the weight image's top-left corner as the origin. */
    size_t        center_coord_x;
    size_t        center_coord_y;
    cl_bitfield   flags;
} cl_weight_desc_qcom;

typedef struct _cl_weight_image_desc_qcom
{
    cl_image_desc           image_desc;
    cl_weight_desc_qcom     weight_desc;
} cl_weight_image_desc_qcom;


/*************************************
 * cl_qcom_protected_context extension *
 *************************************/

#define CL_CONTEXT_PROTECTED_QCOM                    0x40C7
#define CL_MEM_ION_HOST_PTR_PROTECTED_QCOM           0x40C8

#define CL_CONTEXT_PROTECTED_PMEM_QCOM               0x4117
#define CL_MEM_PMEM_HOST_PTR_PROTECTED_QCOM          0x4118

/*************************************
 * cl_qcom_priority_hint extension *
 *************************************/
#define CL_PRIORITY_HINT_NONE_QCOM                   0
typedef cl_uint                                     cl_priority_hint;

#define CL_CONTEXT_PRIORITY_HINT_QCOM               0x40C9

/*cl_priority_hint*/
#define CL_PRIORITY_HINT_HIGH_QCOM                  0x40CA
#define CL_PRIORITY_HINT_NORMAL_QCOM                0x40CB
#define CL_PRIORITY_HINT_LOW_QCOM                   0x40CC

/*************************************
 * cl_recordable_command_queue extension *
 *************************************/

/** Accepted by clGetDeviceInfo */
#define CL_DEVICE_RECORDABLE_QUEUE_MAX_SIZE        0x41DE

/** Flag to enable recordable command queues */
#define CL_QUEUE_RECORDABLE_QCOM                  (1u << 30u)

typedef struct _cl_recording_qcom * cl_recording_qcom;

/** Array element struct used to set kernel arguments */
typedef struct _cl_array_arg_qcom{
    cl_uint dispatch_index;
    cl_uint arg_index;
    size_t arg_size;
    const void *arg_value;
} cl_array_arg_qcom;

typedef struct _cl_array_kernel_exec_info_qcom{
    cl_uint dispatch_index;
    cl_kernel_exec_info param_name;
    size_t param_value_size;
    const void *param_value;
} cl_array_kernel_exec_info_qcom;

/** Used to update a local or global workgroup.  workgroup_size * is used in the same manner as
   the correponding argument in clEnqueueNDRangeKernel */
typedef struct _cl_workgroup_qcom {
    cl_uint dispatch_index;
    const size_t *workgroup_size;
} cl_workgroup_qcom;

typedef struct _cl_offset_qcom
{
    cl_uint dispatch_index;
    size_t offsets[3];
} cl_offset_qcom;


extern CL_API_ENTRY cl_recording_qcom CL_API_CALL
clNewRecordingQCOM(cl_command_queue, cl_int *);
extern CL_API_ENTRY cl_int CL_API_CALL
clEndRecordingQCOM(cl_recording_qcom);
extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseRecordingQCOM(cl_recording_qcom);
extern CL_API_ENTRY cl_int CL_API_CALL
clRetainRecordingQCOM(cl_recording_qcom);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueRecordingQCOM(cl_command_queue    /** command_queue */,
                       cl_recording_qcom   /** recording */,

                       size_t              /** number of recorded args being updated */,
                       const cl_array_arg_qcom * /** recorded arg to update */,

                       size_t               /** Number of global offsets to update */,
                       const cl_offset_qcom * /** Array  offsets to update */,

                       size_t              /** number of global workgroups being updated */,
                       const cl_workgroup_qcom * /** global work group array */,

                       size_t              /** number of local workgroups being updated */,
                       const cl_workgroup_qcom * /** local work size array */,

                       cl_uint             /** num_events_in_wait_list */,
                       const cl_event *    /** event_wait_list */,
                       cl_event *          /** event */);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueRecordingSVMQCOM(cl_command_queue    /** command_queue */,
                          cl_recording_qcom   /** recording */,

                          size_t              /** number of recorded args being updated */,
                          const cl_array_arg_qcom * /** recorded arg to update */,

                          size_t              /** number of recorded SVM args being updated */,
                          const cl_array_arg_qcom * /** recorded SVM arg to update */,

                          size_t               /** Number of global offsets to update */,
                          const cl_offset_qcom * /** Array  offsets to update */,

                          size_t              /** number of global workgroups being updated */,
                          const cl_workgroup_qcom * /** global work group array */,

                          size_t              /** number of local workgroups being updated */,
                          const cl_workgroup_qcom * /** local work size array */,

                          size_t              /** Number of non argument kernel parameters */,
                          const cl_array_kernel_exec_info_qcom * /** Array of non argument kernel parameters to update */,

                          cl_uint             /** num_events_in_wait_list */,
                          const cl_event *    /** event_wait_list */,
                          cl_event *          /** event */);

/**************************
 * cl_qcom_filter_bicubic *
 **************************/

#define CL_FILTER_BICUBIC_QCOM      0x411C

/**************************
 * cl_qcom_dmabuf_host_ptr *
 **************************/

#define CL_MEM_DMABUF_HOST_PTR_QCOM             0x411D
#define CL_MEM_DMABUF_HOST_PTR_PROTECTED_QCOM   0x411E

typedef struct _cl_mem_dmabuf_host_ptr
{
    /* Type of external memory allocation. */
    /* Must be CL_MEM_DMABUF_HOST_PTR_QCOM or CL_MEM_DMABUF_HOST_PTR_PROTECTED_QCOM for dmabuf allocations. */
    cl_mem_ext_host_ptr  ext_host_ptr;

    /* dmabuf file descriptor */
    int                  dmabuf_filedesc;

    /* Host pointer to the dmabuf allocated memory */
    void*                dmabuf_hostptr;

} cl_mem_dmabuf_host_ptr;

/**************************
 * cl_qcom_extended_query_image_info *
 **************************/

#define CL_IMAGE_SIZE_QCOM                        0x411B
#define CL_IMAGE_BASE_ADDRESS_ALIGNMENT_QCOM      0x411F

typedef cl_uint                                   cl_extended_image_info_qcom;

extern CL_API_ENTRY cl_int CL_API_CALL
clQueryImageInfoQCOM(cl_device_id device,
                     cl_mem_flags flags,
                     const cl_image_format * image_format,
                     const cl_image_desc * image_desc,
                     cl_extended_image_info_qcom param_name,
                     size_t                   param_value_size,
                     void                    *param_value,
                     size_t                  *param_value_size_ret);

/**************************
 * cl_qcom_onchip_global_memory *
 **************************/

#define CL_MEM_ONCHIP_GLOBAL_QCOM                                       0x41A2
#define CL_MEM_ONCHIP_GLOBAL_OFFSET_QCOM                                0x41A3
#define CL_DEVICE_ONCHIP_GLOBAL_MEM_SIZE_QCOM                           0x41A4

#ifdef __cplusplus
}
#endif

#endif /* __OPENCL_CL_EXT_QCOM_H */

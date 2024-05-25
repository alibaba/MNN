//
//  OpenCLWrapper.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpenCLWrapper_hpp
#define OpenCLWrapper_hpp

#if defined(WIN32)
#include <Windows.h>
#undef min
#undef max
#undef NO_ERROR
#endif
#include <memory>
#include "core/Macro.h"
#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if !defined(_MSC_VER)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "CL/cl2.hpp"
#pragma GCC diagnostic pop
#else
#include "CL/cl2.hpp"
#endif

#include "CL/cl_ext_qcom.h"

#define MNN_CHECK_NOTNULL(X) MNN_ASSERT(X != NULL)

#define MNN_CHECK_CL_SUCCESS(error, info)                  \
    if (error != CL_SUCCESS) {                       \
        MNN_PRINT("CL ERROR CODE : %d, info:%s \n", (int)error, info); \
    }
#ifdef MNN_USE_LIB_WRAPPER

namespace MNN {

void LoadOpenCLSymbols();
void UnLoadOpenCLSymbols();

class OpenCLSymbols {
public:
    bool LoadOpenCLLibrary();
    bool UnLoadOpenCLLibrary();
    bool isError();
    bool isSvmError();
    bool isPropError();
    bool isQcomError();
    bool isCL1_2Error();
    bool isGlError();
    
    using clGetPlatformIDsFunc        = cl_int (CL_API_CALL *)(cl_uint, cl_platform_id *, cl_uint *);
    using clGetPlatformInfoFunc       = cl_int (CL_API_CALL *)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
    using clBuildProgramFunc          = cl_int (CL_API_CALL *)(cl_program, cl_uint, const cl_device_id *, const char *,
                                          void (CL_CALLBACK *pfn_notify)(cl_program, void *), void *);
    using clEnqueueNDRangeKernelFunc  = cl_int (CL_API_CALL *)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *,
                                                  const size_t *, cl_uint, const cl_event *, cl_event *);
    using clSetKernelArgFunc          = cl_int (CL_API_CALL *)(cl_kernel, cl_uint, size_t, const void *);
    using clRetainMemObjectFunc       = cl_int (CL_API_CALL *)(cl_mem);
    using clReleaseMemObjectFunc      = cl_int (CL_API_CALL *)(cl_mem);
    using clEnqueueUnmapMemObjectFunc = cl_int (CL_API_CALL *)(cl_command_queue, cl_mem, void *, cl_uint, const cl_event *,
                                                   cl_event *);
    using clRetainCommandQueueFunc    = cl_int (CL_API_CALL *)(cl_command_queue command_queue);
    using clCreateContextFunc         = cl_context (CL_API_CALL *)(const cl_context_properties *, cl_uint, const cl_device_id *,
                                               void(CL_CALLBACK *)( // NOLINT(readability/casting)
                                                   const char *, const void *, size_t, void *),
                                               void *, cl_int *);
    using clEnqueueCopyImageFunc = cl_int (CL_API_CALL *)(cl_command_queue,
                       cl_mem,
                       cl_mem,
                       const size_t*,
                       const size_t*,
                       const size_t*,
                       cl_uint,
                       const cl_event*,
                       cl_event*);
    using clEnqueueCopyBufferFunc = cl_int (CL_API_CALL*)(cl_command_queue    /* command_queue */,
                        cl_mem              /* src_buffer */,
                        cl_mem              /* dst_buffer */,
                        size_t              /* src_offset */,
                        size_t              /* dst_offset */,
                        size_t              /* size */,
                        cl_uint             /* num_events_in_wait_list */,
                        const cl_event *    /* event_wait_list */,
                        cl_event *          /* event */);

    using clCreateContextFromTypeFunc = cl_context (CL_API_CALL *)(const cl_context_properties *, cl_device_type,
                                                       void(CL_CALLBACK *)( // NOLINT(readability/casting)
                                                           const char *, const void *, size_t, void *),
                                                       void *, cl_int *);
    using clReleaseContextFunc        = cl_int (CL_API_CALL *)(cl_context);
    using clWaitForEventsFunc         = cl_int (CL_API_CALL *)(cl_uint, const cl_event *);
    using clReleaseEventFunc          = cl_int (CL_API_CALL *)(cl_event);
    using clEnqueueWriteBufferFunc    = cl_int (CL_API_CALL *)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *,
                                                cl_uint, const cl_event *, cl_event *);
    using clEnqueueReadBufferFunc     = cl_int (CL_API_CALL *)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint,
                                               const cl_event *, cl_event *);
    using clEnqueueReadImageFunc     = cl_int (CL_API_CALL *)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
    using clEnqueueWriteImageFunc    = cl_int (CL_API_CALL *)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, size_t, size_t, const void *,
                                        cl_uint, const cl_event *, cl_event * );

    using clGetProgramBuildInfoFunc   = cl_int (CL_API_CALL *)(cl_program, cl_device_id, cl_program_build_info, size_t, void *,
                                                 size_t *);
    using clRetainProgramFunc         = cl_int (CL_API_CALL *)(cl_program program);
    using clEnqueueMapBufferFunc   = void *(CL_API_CALL *)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint,
                                             const cl_event *, cl_event *, cl_int *);
    using clEnqueueMapImageFunc    = void *(CL_API_CALL *)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, const size_t *,
                                            const size_t *, size_t *, size_t *, cl_uint, const cl_event *, cl_event *,
                                            cl_int *);
    using clCreateCommandQueueFunc = cl_command_queue(CL_API_CALL *)( // NOLINT
        cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
    using clReleaseCommandQueueFunc     = cl_int (CL_API_CALL *)(cl_command_queue);
    using clCreateProgramWithBinaryFunc = cl_program (CL_API_CALL *)(cl_context, cl_uint, const cl_device_id *, const size_t *,
                                                         const unsigned char **, cl_int *, cl_int *);
    using clRetainContextFunc           = cl_int (CL_API_CALL *)(cl_context context);
    using clGetContextInfoFunc          = cl_int (CL_API_CALL *)(cl_context, cl_context_info, size_t, void *, size_t *);
    using clReleaseProgramFunc          = cl_int (CL_API_CALL *)(cl_program program);
    using clFlushFunc                   = cl_int (CL_API_CALL *)(cl_command_queue command_queue);
    using clFinishFunc                  = cl_int (CL_API_CALL *)(cl_command_queue command_queue);
    using clGetProgramInfoFunc          = cl_int (CL_API_CALL *)(cl_program, cl_program_info, size_t, void *, size_t *);
    using clCreateKernelFunc            = cl_kernel (CL_API_CALL *)(cl_program, const char *, cl_int *);
    using clRetainKernelFunc            = cl_int (CL_API_CALL *)(cl_kernel kernel);
    using clCreateBufferFunc            = cl_mem (CL_API_CALL *)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
    using clCreateImageFunc             = cl_mem(CL_API_CALL *)(cl_context, cl_mem_flags, const cl_image_format *, const cl_image_desc *, void *, cl_int *);
    using clCreateImage2DFunc           = cl_mem(CL_API_CALL *)(cl_context, // NOLINT
                                                      cl_mem_flags, const cl_image_format *, size_t, size_t, size_t,
                                                      void *, cl_int *);

    using clCreateProgramWithSourceFunc = cl_program (CL_API_CALL *)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
    using clReleaseKernelFunc           = cl_int (CL_API_CALL *)(cl_kernel kernel);
    using clGetDeviceInfoFunc           = cl_int (CL_API_CALL *)(cl_device_id, cl_device_info, size_t, void *, size_t *);
    using clGetDeviceIDsFunc           = cl_int (CL_API_CALL *)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
    using clRetainEventFunc            = cl_int (CL_API_CALL *)(cl_event);
    using clGetKernelWorkGroupInfoFunc = cl_int (CL_API_CALL *)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *,
                                                    size_t *);
    using clGetEventInfoFunc           = cl_int (CL_API_CALL *)(cl_event event, cl_event_info param_name, size_t param_value_size,
                                          void *param_value, size_t *param_value_size_ret);
    using clGetEventProfilingInfoFunc  = cl_int (CL_API_CALL *)(cl_event event, cl_profiling_info param_name,
                                                   size_t param_value_size, void *param_value,
                                                   size_t *param_value_size_ret);
    using clGetMemObjectInfoFunc       = cl_int (CL_API_CALL *)(cl_mem memobj, cl_mem_info param_name,
                                                   size_t param_value_size, void *param_value,
                                                   size_t *param_value_size_ret);
    using clGetImageInfoFunc           = cl_int (CL_API_CALL *)(cl_mem, cl_image_info, size_t, void *, size_t *);
    using clCreateFromGLBufferFunc     = cl_mem (CL_API_CALL *)(cl_context, cl_mem_flags, cl_GLuint, int *);
    using clCreateFromGLTextureFunc     = cl_mem (CL_API_CALL *)(cl_context, cl_mem_flags, cl_GLenum, cl_GLint, cl_GLuint, cl_int*);
    using clEnqueueAcquireGLObjectsFunc = cl_int (CL_API_CALL *)(cl_command_queue, cl_uint, const cl_mem *, cl_uint, const cl_event *, cl_event *);
    using clEnqueueReleaseGLObjectsFunc = cl_int (CL_API_CALL *)(cl_command_queue, cl_uint, const cl_mem *, cl_uint, const cl_event *, cl_event *);
    using clReleaseDeviceFunc = cl_int (CL_API_CALL *)(cl_device_id);
    using clRetainDeviceFunc = cl_int (CL_API_CALL *)(cl_device_id);

    // opencl 2.0 get sub group info and wave size.
    using clCreateCommandQueueWithPropertiesFunc = cl_command_queue (CL_API_CALL *)(cl_context, cl_device_id,
                                                    const cl_queue_properties *, cl_int *);
    using clSVMAllocFunc = void *(CL_API_CALL *)(cl_context, cl_mem_flags, size_t size, cl_uint);
    using clSVMFreeFunc = void (CL_API_CALL *)(cl_context, void *);
    using clEnqueueSVMMapFunc = cl_int (CL_API_CALL *)(cl_command_queue, cl_bool, cl_map_flags,
                                           void *, size_t, cl_uint, const cl_event *, cl_event *);
    using clEnqueueSVMUnmapFunc = cl_int (CL_API_CALL *)(cl_command_queue, void *, cl_uint,
                                             const cl_event *, cl_event *);
    using clSetKernelArgSVMPointerFunc = cl_int (CL_API_CALL *)(cl_kernel, cl_uint, const void *);
    
    using clNewRecordingQCOMFunc = cl_recording_qcom(CL_API_CALL *)(cl_command_queue, cl_int *);
    using clEndRecordingQCOMFunc = cl_int (CL_API_CALL *)(cl_recording_qcom);
    using clReleaseRecordingQCOMFunc = cl_int (CL_API_CALL *)(cl_recording_qcom);
    using clRetainRecordingQCOMFunc = cl_int (CL_API_CALL *)(cl_recording_qcom);
    using clEnqueueRecordingQCOMFunc = cl_int (CL_API_CALL *)(cl_command_queue, cl_recording_qcom, size_t, const cl_array_arg_qcom*, size_t, const cl_offset_qcom*,
                                                  size_t, const cl_workgroup_qcom*, size_t, const cl_workgroup_qcom*, cl_uint, const cl_event*, cl_event*);
    using clEnqueueRecordingSVMQCOMFunc = cl_int (CL_API_CALL *)(cl_command_queue, cl_recording_qcom, size_t, const cl_array_arg_qcom*, size_t, const cl_array_arg_qcom*,
                                                     size_t, const cl_offset_qcom*, size_t, const cl_workgroup_qcom*, size_t, const cl_workgroup_qcom*,
                                                     size_t, const cl_array_kernel_exec_info_qcom*, cl_uint, const cl_event*, cl_event*);
    
#define MNN_CL_DEFINE_FUNC_PTR(func) func##Func func = nullptr

    MNN_CL_DEFINE_FUNC_PTR(clGetPlatformIDs);
    MNN_CL_DEFINE_FUNC_PTR(clGetPlatformInfo);
    MNN_CL_DEFINE_FUNC_PTR(clBuildProgram);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueNDRangeKernel);
    MNN_CL_DEFINE_FUNC_PTR(clSetKernelArg);
    MNN_CL_DEFINE_FUNC_PTR(clReleaseKernel);
    MNN_CL_DEFINE_FUNC_PTR(clCreateProgramWithSource);
    MNN_CL_DEFINE_FUNC_PTR(clCreateBuffer);
    MNN_CL_DEFINE_FUNC_PTR(clCreateImage);
    MNN_CL_DEFINE_FUNC_PTR(clCreateImage2D);
    MNN_CL_DEFINE_FUNC_PTR(clRetainKernel);
    MNN_CL_DEFINE_FUNC_PTR(clCreateKernel);
    MNN_CL_DEFINE_FUNC_PTR(clGetProgramInfo);
    MNN_CL_DEFINE_FUNC_PTR(clFlush);
    MNN_CL_DEFINE_FUNC_PTR(clFinish);
    MNN_CL_DEFINE_FUNC_PTR(clReleaseProgram);
    MNN_CL_DEFINE_FUNC_PTR(clRetainContext);
    MNN_CL_DEFINE_FUNC_PTR(clGetContextInfo);
    MNN_CL_DEFINE_FUNC_PTR(clCreateProgramWithBinary);
    MNN_CL_DEFINE_FUNC_PTR(clCreateCommandQueue);
    MNN_CL_DEFINE_FUNC_PTR(clReleaseCommandQueue);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueCopyBuffer);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueMapBuffer);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueMapImage);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueCopyImage);
    MNN_CL_DEFINE_FUNC_PTR(clRetainProgram);
    MNN_CL_DEFINE_FUNC_PTR(clGetProgramBuildInfo);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueReadBuffer);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueWriteBuffer);
    MNN_CL_DEFINE_FUNC_PTR(clWaitForEvents);
    MNN_CL_DEFINE_FUNC_PTR(clReleaseEvent);
    MNN_CL_DEFINE_FUNC_PTR(clCreateContext);
    MNN_CL_DEFINE_FUNC_PTR(clCreateContextFromType);
    MNN_CL_DEFINE_FUNC_PTR(clReleaseContext);
    MNN_CL_DEFINE_FUNC_PTR(clRetainCommandQueue);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueUnmapMemObject);
    MNN_CL_DEFINE_FUNC_PTR(clRetainMemObject);
    MNN_CL_DEFINE_FUNC_PTR(clReleaseMemObject);
    MNN_CL_DEFINE_FUNC_PTR(clGetDeviceInfo);
    MNN_CL_DEFINE_FUNC_PTR(clGetDeviceIDs);
    MNN_CL_DEFINE_FUNC_PTR(clRetainEvent);
    MNN_CL_DEFINE_FUNC_PTR(clGetKernelWorkGroupInfo);
    MNN_CL_DEFINE_FUNC_PTR(clGetEventInfo);
    MNN_CL_DEFINE_FUNC_PTR(clGetEventProfilingInfo);
    MNN_CL_DEFINE_FUNC_PTR(clGetMemObjectInfo);
    MNN_CL_DEFINE_FUNC_PTR(clGetImageInfo);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueReadImage);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueWriteImage);
    MNN_CL_DEFINE_FUNC_PTR(clCreateFromGLBuffer);
    MNN_CL_DEFINE_FUNC_PTR(clCreateFromGLTexture);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueAcquireGLObjects);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueReleaseGLObjects);
    MNN_CL_DEFINE_FUNC_PTR(clRetainDevice);
    MNN_CL_DEFINE_FUNC_PTR(clReleaseDevice);
    
    MNN_CL_DEFINE_FUNC_PTR(clCreateCommandQueueWithProperties);
    MNN_CL_DEFINE_FUNC_PTR(clSVMAlloc);
    MNN_CL_DEFINE_FUNC_PTR(clSVMFree);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueSVMMap);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueSVMUnmap);
    MNN_CL_DEFINE_FUNC_PTR(clSetKernelArgSVMPointer);
    
    MNN_CL_DEFINE_FUNC_PTR(clNewRecordingQCOM);
    MNN_CL_DEFINE_FUNC_PTR(clEndRecordingQCOM);
    MNN_CL_DEFINE_FUNC_PTR(clReleaseRecordingQCOM);
    MNN_CL_DEFINE_FUNC_PTR(clRetainRecordingQCOM);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueRecordingQCOM);
    MNN_CL_DEFINE_FUNC_PTR(clEnqueueRecordingSVMQCOM);

#undef MNN_CL_DEFINE_FUNC_PTR

private:
    bool LoadLibraryFromPath(const std::string &path);
#if defined(WIN32)
    HMODULE handle_ = nullptr;
#else
    void *handle_ = nullptr;
#endif
    bool mIsError{false};
    bool mSvmError{false};
    bool mPropError{false};
    bool mQcomError{false};
    bool mCL_12Error{false};
    bool mGlError{false};
};

class OpenCLSymbolsOperator {
public:
    static OpenCLSymbolsOperator *createOpenCLSymbolsOperatorSingleInstance();

    static OpenCLSymbols *getOpenclSymbolsPtr();
    OpenCLSymbolsOperator();
    ~OpenCLSymbolsOperator();

private:
    OpenCLSymbolsOperator(const OpenCLSymbolsOperator &) = delete;
    OpenCLSymbolsOperator &operator=(const OpenCLSymbolsOperator &) = delete;

    static std::shared_ptr<OpenCLSymbols> gOpenclSymbols;
};

} // namespace MNN
#endif
#endif  /* OpenCLWrapper_hpp */

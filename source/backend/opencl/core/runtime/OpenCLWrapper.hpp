//
//  OpenCLWrapper.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef OpenCLWrapper_hpp
#define OpenCLWrapper_hpp


#include <memory>
#include "Macro.h"
#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 110
#define CL_HPP_MINIMUM_OPENCL_VERSION 110

// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Weffc++"
// #pragma GCC diagnostic ignored "-Wignored-qualifiers"
#include "CL/cl2.hpp"
// #pragma GCC diagnostic pop

#define MNN_CHECK_NOTNULL(X) MNN_ASSERT(X != NULL)

#define MNN_CHECK_CL_SUCCESS(error)                  \
    if (error != CL_SUCCESS) {                       \
        MNN_PRINT("ERROR CODE : %d \n", (int)error); \
    }
#ifdef MNN_USE_OPENCL_WRAPPER

namespace MNN {

void LoadOpenCLSymbols();
void UnLoadOpenCLSymbols();

class OpenCLSymbols {
public:
    bool LoadOpenCLLibrary();
    bool UnLoadOpenCLLibrary();
    bool isError();
    using clGetPlatformIDsFunc        = cl_int (*)(cl_uint, cl_platform_id *, cl_uint *);
    using clGetPlatformInfoFunc       = cl_int (*)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
    using clBuildProgramFunc          = cl_int (*)(cl_program, cl_uint, const cl_device_id *, const char *,
                                          void (*pfn_notify)(cl_program, void *), void *);
    using clEnqueueNDRangeKernelFunc  = cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *,
                                                  const size_t *, cl_uint, const cl_event *, cl_event *);
    using clSetKernelArgFunc          = cl_int (*)(cl_kernel, cl_uint, size_t, const void *);
    using clRetainMemObjectFunc       = cl_int (*)(cl_mem);
    using clReleaseMemObjectFunc      = cl_int (*)(cl_mem);
    using clEnqueueUnmapMemObjectFunc = cl_int (*)(cl_command_queue, cl_mem, void *, cl_uint, const cl_event *,
                                                   cl_event *);
    using clRetainCommandQueueFunc    = cl_int (*)(cl_command_queue command_queue);
    using clCreateContextFunc         = cl_context (*)(const cl_context_properties *, cl_uint, const cl_device_id *,
                                               void(CL_CALLBACK *)( // NOLINT(readability/casting)
                                                   const char *, const void *, size_t, void *),
                                               void *, cl_int *);
    using clEnqueueCopyImageFunc = cl_int (*)(cl_command_queue,
                       cl_mem,
                       cl_mem,
                       const size_t*,
                       const size_t*,
                       const size_t*,
                       cl_uint,
                       const cl_event*,
                       cl_event*);

    using clCreateContextFromTypeFunc = cl_context (*)(const cl_context_properties *, cl_device_type,
                                                       void(CL_CALLBACK *)( // NOLINT(readability/casting)
                                                           const char *, const void *, size_t, void *),
                                                       void *, cl_int *);
    using clReleaseContextFunc        = cl_int (*)(cl_context);
    using clWaitForEventsFunc         = cl_int (*)(cl_uint, const cl_event *);
    using clReleaseEventFunc          = cl_int (*)(cl_event);
    using clEnqueueWriteBufferFunc    = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *,
                                                cl_uint, const cl_event *, cl_event *);
    using clEnqueueReadBufferFunc     = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint,
                                               const cl_event *, cl_event *);
    using clGetProgramBuildInfoFunc   = cl_int (*)(cl_program, cl_device_id, cl_program_build_info, size_t, void *,
                                                 size_t *);
    using clRetainProgramFunc         = cl_int (*)(cl_program program);
    using clEnqueueMapBufferFunc   = void *(*)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint,
                                             const cl_event *, cl_event *, cl_int *);
    using clEnqueueMapImageFunc    = void *(*)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, const size_t *,
                                            const size_t *, size_t *, size_t *, cl_uint, const cl_event *, cl_event *,
                                            cl_int *);
    using clCreateCommandQueueFunc = cl_command_queue(CL_API_CALL *)( // NOLINT
        cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
    using clReleaseCommandQueueFunc     = cl_int (*)(cl_command_queue);
    using clCreateProgramWithBinaryFunc = cl_program (*)(cl_context, cl_uint, const cl_device_id *, const size_t *,
                                                         const unsigned char **, cl_int *, cl_int *);
    using clRetainContextFunc           = cl_int (*)(cl_context context);
    using clGetContextInfoFunc          = cl_int (*)(cl_context, cl_context_info, size_t, void *, size_t *);
    using clReleaseProgramFunc          = cl_int (*)(cl_program program);
    using clFlushFunc                   = cl_int (*)(cl_command_queue command_queue);
    using clFinishFunc                  = cl_int (*)(cl_command_queue command_queue);
    using clGetProgramInfoFunc          = cl_int (*)(cl_program, cl_program_info, size_t, void *, size_t *);
    using clCreateKernelFunc            = cl_kernel (*)(cl_program, const char *, cl_int *);
    using clRetainKernelFunc            = cl_int (*)(cl_kernel kernel);
    using clCreateBufferFunc            = cl_mem (*)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
    using clCreateImage2DFunc           = cl_mem(CL_API_CALL *)(cl_context, // NOLINT
                                                      cl_mem_flags, const cl_image_format *, size_t, size_t, size_t,
                                                      void *, cl_int *);
    using clCreateImageFunc = cl_mem (*)(cl_context, cl_mem_flags, const cl_image_format *, const cl_image_desc *,
                                         void *, cl_int *);
    using clCreateProgramWithSourceFunc = cl_program (*)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
    using clReleaseKernelFunc           = cl_int (*)(cl_kernel kernel);
    using clGetDeviceInfoFunc           = cl_int (*)(cl_device_id, cl_device_info, size_t, void *, size_t *);
    using clGetDeviceIDsFunc           = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
    using clRetainDeviceFunc           = cl_int (*)(cl_device_id);
    using clReleaseDeviceFunc          = cl_int (*)(cl_device_id);
    using clRetainEventFunc            = cl_int (*)(cl_event);
    using clGetKernelWorkGroupInfoFunc = cl_int (*)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void *,
                                                    size_t *);
    using clGetEventInfoFunc           = cl_int (*)(cl_event event, cl_event_info param_name, size_t param_value_size,
                                          void *param_value, size_t *param_value_size_ret);
    using clGetEventProfilingInfoFunc  = cl_int (*)(cl_event event, cl_profiling_info param_name,
                                                   size_t param_value_size, void *param_value,
                                                   size_t *param_value_size_ret);
    using clGetImageInfoFunc           = cl_int (*)(cl_mem, cl_image_info, size_t, void *, size_t *);

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
    MNN_CL_DEFINE_FUNC_PTR(clRetainDevice);
    MNN_CL_DEFINE_FUNC_PTR(clReleaseDevice);
    MNN_CL_DEFINE_FUNC_PTR(clRetainEvent);
    MNN_CL_DEFINE_FUNC_PTR(clGetKernelWorkGroupInfo);
    MNN_CL_DEFINE_FUNC_PTR(clGetEventInfo);
    MNN_CL_DEFINE_FUNC_PTR(clGetEventProfilingInfo);
    MNN_CL_DEFINE_FUNC_PTR(clGetImageInfo);

#undef MNN_CL_DEFINE_FUNC_PTR

private:
    bool LoadLibraryFromPath(const std::string &path);
    void *handle_ = nullptr;
    bool mIsError{false};
};

class OpenCLSymbolsOperator {
public:
    static OpenCLSymbolsOperator *createOpenCLSymbolsOperatorSingleInstance() {
        static OpenCLSymbolsOperator symbols_operator;
        return &symbols_operator;
    }

    static OpenCLSymbols *getOpenclSymbolsPtr();

private:
    OpenCLSymbolsOperator();
    ~OpenCLSymbolsOperator();
    OpenCLSymbolsOperator(const OpenCLSymbolsOperator &) = delete;
    OpenCLSymbolsOperator &operator=(const OpenCLSymbolsOperator &) = delete;

    static std::shared_ptr<OpenCLSymbols> gOpenclSymbols;
};

} // namespace MNN
#endif
#endif  /* OpenCLWrapper_hpp */

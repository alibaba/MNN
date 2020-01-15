//
//  OpenCLWrapper.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_USE_OPENCL_WRAPPER
#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"
#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>

namespace MNN {
static const std::vector<std::string> gOpencl_library_paths = {

#if defined(__APPLE__) || defined(__MACOSX)
    "libOpenCL.so", "/System/Library/Frameworks/OpenCL.framework/OpenCL"
#elif defined(__ANDROID__)
    "libOpenCL.so",
    "libGLES_mali.so",
    "libmali.so",
#if defined(__aarch64__)
    // Qualcomm Adreno
    "/system/vendor/lib64/libOpenCL.so",
    "/system/lib64/libOpenCL.so",
    // Mali
    "/system/vendor/lib64/egl/libGLES_mali.so",
    "/system/lib64/egl/libGLES_mali.so",
#else
    // Qualcomm Adreno
    "/system/vendor/lib/libOpenCL.so", "/system/lib/libOpenCL.so",
    // Mali
    "/system/vendor/lib/egl/libGLES_mali.so", "/system/lib/egl/libGLES_mali.so",
    // other
    "/system/vendor/lib/libPVROCL.so", "/data/data/org.pocl.libs/files/lib/libpocl.so"
#endif
#elif defined(__linux__)
    "/usr/lib/libOpenCL.so",
    "/usr/local/lib/libOpenCL.so",
    "/usr/local/lib/libpocl.so",
    "/usr/lib64/libOpenCL.so",
    "/usr/lib32/libOpenCL.so",
    "libOpenCL.so"
#endif
};
bool OpenCLSymbols::LoadOpenCLLibrary() {
    if (handle_ != nullptr) {
        return true;
    }
    for (const auto &opencl_lib : gOpencl_library_paths) {
        if (LoadLibraryFromPath(opencl_lib)) {
            return true;
        }
    }
    return false;
}

bool OpenCLSymbols::UnLoadOpenCLLibrary() {
    if (handle_ != nullptr) {
        if (dlclose(handle_) != 0) {
            return false;
        }
        handle_ = nullptr;
        return true;
    }
    return true;
}

bool OpenCLSymbols::isError() {
    return mIsError;
}

bool OpenCLSymbols::LoadLibraryFromPath(const std::string &library_path) {
    handle_ = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
        return false;
    }

#define MNN_LOAD_FUNCTION_PTR(func_name) func_name = reinterpret_cast<func_name##Func>(dlsym(handle_, #func_name)); \
    if(func_name == nullptr){ \
        mIsError = true; \
    }

    MNN_LOAD_FUNCTION_PTR(clGetPlatformIDs);
    MNN_LOAD_FUNCTION_PTR(clGetPlatformInfo);
    MNN_LOAD_FUNCTION_PTR(clBuildProgram);
    MNN_LOAD_FUNCTION_PTR(clEnqueueNDRangeKernel);
    MNN_LOAD_FUNCTION_PTR(clSetKernelArg);
    MNN_LOAD_FUNCTION_PTR(clReleaseKernel);
    MNN_LOAD_FUNCTION_PTR(clCreateProgramWithSource);
    MNN_LOAD_FUNCTION_PTR(clCreateBuffer);
    //MNN_LOAD_FUNCTION_PTR(clCreateImage);
    MNN_LOAD_FUNCTION_PTR(clCreateImage2D);
    MNN_LOAD_FUNCTION_PTR(clRetainKernel);
    MNN_LOAD_FUNCTION_PTR(clCreateKernel);
    MNN_LOAD_FUNCTION_PTR(clGetProgramInfo);
    MNN_LOAD_FUNCTION_PTR(clFlush);
    MNN_LOAD_FUNCTION_PTR(clFinish);
    MNN_LOAD_FUNCTION_PTR(clReleaseProgram);
    MNN_LOAD_FUNCTION_PTR(clRetainContext);
    MNN_LOAD_FUNCTION_PTR(clGetContextInfo);
    MNN_LOAD_FUNCTION_PTR(clCreateProgramWithBinary);
    MNN_LOAD_FUNCTION_PTR(clCreateCommandQueue);
    MNN_LOAD_FUNCTION_PTR(clReleaseCommandQueue);
    MNN_LOAD_FUNCTION_PTR(clEnqueueMapBuffer);
    MNN_LOAD_FUNCTION_PTR(clEnqueueMapImage);
    MNN_LOAD_FUNCTION_PTR(clRetainProgram);
    MNN_LOAD_FUNCTION_PTR(clGetProgramBuildInfo);
    MNN_LOAD_FUNCTION_PTR(clEnqueueReadBuffer);
    MNN_LOAD_FUNCTION_PTR(clEnqueueWriteBuffer);
    MNN_LOAD_FUNCTION_PTR(clWaitForEvents);
    MNN_LOAD_FUNCTION_PTR(clReleaseEvent);
    MNN_LOAD_FUNCTION_PTR(clCreateContext);
    MNN_LOAD_FUNCTION_PTR(clCreateContextFromType);
    MNN_LOAD_FUNCTION_PTR(clReleaseContext);
    MNN_LOAD_FUNCTION_PTR(clRetainCommandQueue);
    MNN_LOAD_FUNCTION_PTR(clEnqueueUnmapMemObject);
    MNN_LOAD_FUNCTION_PTR(clRetainMemObject);
    MNN_LOAD_FUNCTION_PTR(clReleaseMemObject);
    MNN_LOAD_FUNCTION_PTR(clGetDeviceInfo);
    MNN_LOAD_FUNCTION_PTR(clGetDeviceIDs);
    //MNN_LOAD_FUNCTION_PTR(clRetainDevice);
    //MNN_LOAD_FUNCTION_PTR(clReleaseDevice);
    MNN_LOAD_FUNCTION_PTR(clRetainEvent);
    MNN_LOAD_FUNCTION_PTR(clGetKernelWorkGroupInfo);
    MNN_LOAD_FUNCTION_PTR(clGetEventInfo);
    MNN_LOAD_FUNCTION_PTR(clGetEventProfilingInfo);
    MNN_LOAD_FUNCTION_PTR(clGetImageInfo);
    MNN_LOAD_FUNCTION_PTR(clEnqueueCopyImage);
    MNN_LOAD_FUNCTION_PTR(clEnqueueReadImage);
    MNN_LOAD_FUNCTION_PTR(clEnqueueWriteImage);
#undef MNN_LOAD_FUNCTION_PTR

    return true;
}

std::shared_ptr<OpenCLSymbols> OpenCLSymbolsOperator::gOpenclSymbols;

OpenCLSymbols *OpenCLSymbolsOperator::getOpenclSymbolsPtr() {
    return gOpenclSymbols.get();
}

OpenCLSymbolsOperator::OpenCLSymbolsOperator() {
#ifdef LOG_VERBOSE
    MNN_PRINT("start OpenCLSymbolsOperator !\n");
#endif
    if (gOpenclSymbols.get() == nullptr) {
        gOpenclSymbols.reset(new OpenCLSymbols());
    } else {
#ifdef LOG_VERBOSE
        MNN_PRINT(" OpenCLSymbols already now !\n");
#endif
    }

    if (false == gOpenclSymbols->LoadOpenCLLibrary()) {
        gOpenclSymbols.reset();
    }
#ifdef LOG_VERBOSE
    MNN_PRINT("end OpenCLSymbolsOperator !\n");
#endif
}

OpenCLSymbolsOperator::~OpenCLSymbolsOperator() {
#ifdef LOG_VERBOSE
    MNN_PRINT("start ~OpenCLSymbolsOperator !\n");
#endif
    if (nullptr == gOpenclSymbols.get()) {
        return;
    }
    gOpenclSymbols.get()->UnLoadOpenCLLibrary();
#ifdef LOG_VERBOSE
    MNN_PRINT("end ~OpenCLSymbolsOperator !\n");
#endif
}

} // namespace MNN

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetPlatformIDs;
    MNN_CHECK_NOTNULL(func);
    return func(num_entries, platforms, num_platforms);
}

cl_int clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size,
                         void *param_value, size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetPlatformInfo;
    MNN_CHECK_NOTNULL(func);
    return func(platform, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices,
                      cl_uint *num_devices) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetDeviceIDs;
    MNN_CHECK_NOTNULL(func);
    return func(platform, device_type, num_entries, devices, num_devices);
}

cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value,
                       size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetDeviceInfo;
    MNN_CHECK_NOTNULL(func);
    return func(device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clRetainDevice(cl_device_id device) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clRetainDevice;
    MNN_CHECK_NOTNULL(func);
    return func(device);
}

cl_int clReleaseDevice(cl_device_id device) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clReleaseDevice;
    MNN_CHECK_NOTNULL(func);
    return func(device);
}

cl_context clCreateContext(const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices,
                           void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *), void *user_data,
                           cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateContext;
    MNN_CHECK_NOTNULL(func);
    return func(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}

cl_context clCreateContextFromType(const cl_context_properties *properties, cl_device_type device_type,
                                   void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
                                   void *user_data, cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateContextFromType;
    MNN_CHECK_NOTNULL(func);
    return func(properties, device_type, pfn_notify, user_data, errcode_ret);
}

cl_int clRetainContext(cl_context context) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clRetainContext;
    MNN_CHECK_NOTNULL(func);
    return func(context);
}

cl_int clReleaseContext(cl_context context) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clReleaseContext;
    MNN_CHECK_NOTNULL(func);
    return func(context);
}

cl_int clGetContextInfo(cl_context context, cl_context_info param_name, size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetContextInfo;
    MNN_CHECK_NOTNULL(func);
    return func(context, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths,
                                     cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateProgramWithSource;
    MNN_CHECK_NOTNULL(func);
    return func(context, count, strings, lengths, errcode_ret);
}

cl_int clGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetProgramInfo;
    MNN_CHECK_NOTNULL(func);
    return func(program, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name,
                             size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetProgramBuildInfo;
    MNN_CHECK_NOTNULL(func);
    return func(program, device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clRetainProgram(cl_program program) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clRetainProgram;
    MNN_CHECK_NOTNULL(func);
    return func(program);
}

cl_int clReleaseProgram(cl_program program) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clReleaseProgram;
    MNN_CHECK_NOTNULL(func);
    return func(program);
}

cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options,
                      void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data), void *user_data) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clBuildProgram;
    MNN_CHECK_NOTNULL(func);
    return func(program, num_devices, device_list, options, pfn_notify, user_data);
}

cl_kernel clCreateKernel(cl_program program, const char *kernelName, cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateKernel;
    MNN_CHECK_NOTNULL(func);
    return func(program, kernelName, errcode_ret);
}

cl_int clRetainKernel(cl_kernel kernel) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clRetainKernel;
    MNN_CHECK_NOTNULL(func);
    return func(kernel);
}

cl_int clReleaseKernel(cl_kernel kernel) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clReleaseKernel;
    MNN_CHECK_NOTNULL(func);
    return func(kernel);
}

cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clSetKernelArg;
    MNN_CHECK_NOTNULL(func);
    return func(kernel, arg_index, arg_size, arg_value);
}

cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateBuffer;
    MNN_CHECK_NOTNULL(func);
    return func(context, flags, size, host_ptr, errcode_ret);
}

cl_mem clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format *image_format,
                     const cl_image_desc *image_desc, void *host_ptr, cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateImage;
    MNN_CHECK_NOTNULL(func);
    return func(context, flags, image_format, image_desc, host_ptr, errcode_ret);
}

cl_int clRetainMemObject(cl_mem memobj) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clRetainMemObject;
    MNN_CHECK_NOTNULL(func);
    return func(memobj);
}

cl_int clReleaseMemObject(cl_mem memobj) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clReleaseMemObject;
    MNN_CHECK_NOTNULL(func);
    return func(memobj);
}

cl_int clGetImageInfo(cl_mem image, cl_image_info param_name, size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetImageInfo;
    MNN_CHECK_NOTNULL(func);
    return func(image, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clRetainCommandQueue(cl_command_queue command_queue) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clRetainCommandQueue;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue);
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clReleaseCommandQueue;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue);
}

cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset,
                           size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                           cl_event *event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueReadBuffer;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list,
                event);
}

cl_int clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset,
                            size_t size, const void *ptr, cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list, cl_event *event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueWriteBuffer;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list,
                event);
}

cl_int clEnqueueReadImage(cl_command_queue command_queue, cl_mem cl_image, cl_bool is_block, const size_t * origin, const size_t * region, size_t row_pitch,
                                       size_t slice_pitch, void * ptr, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event) CL_API_SUFFIX__VERSION_1_0 {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueReadImage;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, cl_image, is_block, origin, region, row_pitch, slice_pitch, ptr, num_events_in_wait_list, event_wait_list,
                event);
}

cl_int clEnqueueWriteImage(cl_command_queue command_queue, cl_mem cl_image, cl_bool is_block, const size_t * origin, const size_t * region, size_t row_pitch,
                                       size_t slice_pitch, const void * ptr, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueWriteImage;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, cl_image, is_block, origin, region, row_pitch, slice_pitch, ptr, num_events_in_wait_list, event_wait_list,
                event);
}

void *clEnqueueMapBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_map, cl_map_flags map_flags,
                         size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                         cl_event *event, cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueMapBuffer;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list, event_wait_list,
                event, errcode_ret);
}

void *clEnqueueMapImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_map, cl_map_flags map_flags,
                        const size_t *origin, const size_t *region, size_t *image_row_pitch, size_t *image_slice_pitch,
                        cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event,
                        cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueMapImage;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, image, blocking_map, map_flags, origin, region, image_row_pitch, image_slice_pitch,
                num_events_in_wait_list, event_wait_list, event, errcode_ret);
}

cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue, cl_mem memobj, void *mapped_ptr,
                               cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueUnmapMemObject;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
}

cl_int clGetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name,
                                size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetKernelWorkGroupInfo;
    MNN_CHECK_NOTNULL(func);
    return func(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetEventProfilingInfo;
    MNN_CHECK_NOTNULL(func);
    return func(event, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
                              const size_t *global_work_offset, const size_t *global_work_size,
                              const size_t *local_work_size, cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list, cl_event *event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueNDRangeKernel;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size,
                num_events_in_wait_list, event_wait_list, event);
}

cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clWaitForEvents;
    MNN_CHECK_NOTNULL(func);
    return func(num_events, event_list);
}

cl_int clRetainEvent(cl_event event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clRetainEvent;
    MNN_CHECK_NOTNULL(func);
    return func(event);
}

cl_int clReleaseEvent(cl_event event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clReleaseEvent;
    MNN_CHECK_NOTNULL(func);
    return func(event);
}

cl_int clGetEventInfo(cl_event event, cl_event_info param_name, size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetEventInfo;
    MNN_CHECK_NOTNULL(func);
    return func(event, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clFlush(cl_command_queue command_queue) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clFlush;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue);
}

cl_int clFinish(cl_command_queue command_queue) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clFinish;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue);
}

cl_mem clCreateImage2D(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, size_t imageWidth,
                       size_t imageHeight, size_t image_row_pitch, void *host_ptr, cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateImage2D;
    MNN_CHECK_NOTNULL(func);
    return func(context, flags, image_format, imageWidth, imageHeight, image_row_pitch, host_ptr, errcode_ret);
}

cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties,
                                      cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateCommandQueue;
    MNN_CHECK_NOTNULL(func);
    return func(context, device, properties, errcode_ret);
}
cl_int clEnqueueCopyImage(cl_command_queue queue,
                   cl_mem src_image,
                   cl_mem dst_image,
                   const size_t * src_origin,
                   const size_t * dst_origin,
                   const size_t * region,
                   cl_uint        num_events_in_wait_list,
                   const cl_event * event_wait_list ,
                   cl_event *  event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueCopyImage;
    MNN_CHECK_NOTNULL(func);
    return func(queue, src_image, dst_image, src_origin, dst_origin, region, num_events_in_wait_list, event_wait_list, event);
}
#endif

//
//  OpenCLWrapper.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/opencl/core/runtime/OpenCLWrapper.hpp"
#ifdef WIN32
#include <libloaderapi.h>
#else
#include <dlfcn.h>
#endif
#include <memory>
#include <string>
#include <vector>
#include <mutex>

#ifdef MNN_USE_LIB_WRAPPER

namespace MNN {
bool OpenCLSymbols::LoadOpenCLLibrary() {
    if (handle_ != nullptr) {
        return true;
    }
    static const std::vector<std::string> gOpencl_library_paths = {

    #if defined(__APPLE__) || defined(__MACOSX)
        "libOpenCL.so", "/System/Library/Frameworks/OpenCL.framework/OpenCL"
    #elif defined(__ANDROID__)
        "libOpenCL.so",
        "libGLES_mali.so",
        "libmali.so",
        "libOpenCL-pixel.so",
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
    /*
     *  0: System32, 1: SysWOW64
     *  --------------------------------------
     *  | Real CPU /          |  x64  |  x86  |
     *  |        / Target CPU |       |       |
     *  --------------------------------------
     *  |         x64         | 0 / 1 |   1   |
     *  --------------------------------------
     *  |         x86         | Error |   0   |
     *  --------------------------------------
     *  0 / 1: 0 if OpenCL.dll (System32, 64bit on x64), otherwise 1 (SysWOW64, 32bit compatible on 64bit OS)
     */
    #elif defined(_WIN64)
        "C:/Windows/System32/OpenCL.dll",
        "C:/Windows/SysWOW64/OpenCL.dll"
    #elif defined(_WIN32)
        "C:/Windows/SysWOW64/OpenCL.dll",
        "C:/Windows/System32/OpenCL.dll"
    #endif
    };

    for (const auto &opencl_lib : gOpencl_library_paths) {
        if (LoadLibraryFromPath(opencl_lib)) {
            return true;
        }
    }
    return false;
}

bool OpenCLSymbols::UnLoadOpenCLLibrary() {
    if (handle_ != nullptr) {
#if defined(WIN32)
        if (FreeLibrary(handle_) == 0) {
#else
        if (dlclose(handle_) != 0) {
#endif
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

bool OpenCLSymbols::isSvmError() {
    return mSvmError;
}

bool OpenCLSymbols::isPropError() {
    return mPropError;
}
    
bool OpenCLSymbols::LoadLibraryFromPath(const std::string &library_path) {
#if defined(WIN32)
    handle_ = LoadLibraryA(library_path.c_str());
    if (handle_ == nullptr) {
        return false;
    }
#define MNN_LOAD_FUNCTION_PTR(func_name) func_name = reinterpret_cast<func_name##Func>(GetProcAddress(handle_, #func_name)); \
    if(func_name == nullptr){ \
        mIsError = true; \
    }
    
#define MNN_LOAD_SVM_PTR(func_name) func_name = reinterpret_cast<func_name##Func>(GetProcAddress(handle_, #func_name)); \
    if(func_name == nullptr){ \
        mSvmError = true; \
    }
   
#define MNN_LOAD_PROP_PTR(func_name) func_name = reinterpret_cast<func_name##Func>(GetProcAddress(handle_, #func_name)); \
    if(func_name == nullptr){ \
        mPropError = true; \
    }
    
#else
    handle_ = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
        return false;
    }
    
    typedef void* (*loadOpenCLPointerFunc)(const char* name);
    typedef void (*enableOpenCLFunc)();
    loadOpenCLPointerFunc loadOpenCLPointer = nullptr;
    enableOpenCLFunc enableOpenCL = reinterpret_cast<enableOpenCLFunc>(dlsym(handle_, "enableOpenCL"));
    if(enableOpenCL != nullptr){
        enableOpenCL();
        loadOpenCLPointer = reinterpret_cast<loadOpenCLPointerFunc>(dlsym(handle_, "loadOpenCLPointer"));
    }
#define MNN_LOAD_FUNCTION_PTR(func_name) func_name = reinterpret_cast<func_name##Func>(dlsym(handle_, #func_name)); \
    if(func_name == nullptr && loadOpenCLPointer != nullptr){ \
        func_name = reinterpret_cast<func_name##Func>(loadOpenCLPointer(#func_name)); \
    } \
    if(func_name == nullptr){ \
        mIsError = true; \
    }
    
#define MNN_LOAD_SVM_PTR(func_name) func_name = reinterpret_cast<func_name##Func>(dlsym(handle_, #func_name)); \
    if(func_name == nullptr && loadOpenCLPointer != nullptr){ \
        func_name = reinterpret_cast<func_name##Func>(loadOpenCLPointer(#func_name)); \
    } \
    if(func_name == nullptr){ \
        mSvmError = true; \
    }
    
#define MNN_LOAD_PROP_PTR(func_name) func_name = reinterpret_cast<func_name##Func>(dlsym(handle_, #func_name)); \
    if(func_name == nullptr && loadOpenCLPointer != nullptr){ \
        func_name = reinterpret_cast<func_name##Func>(loadOpenCLPointer(#func_name)); \
    } \
    if(func_name == nullptr){ \
        mPropError = true; \
    }
#endif

    MNN_LOAD_FUNCTION_PTR(clGetPlatformIDs);
    MNN_LOAD_FUNCTION_PTR(clGetPlatformInfo);
    MNN_LOAD_FUNCTION_PTR(clBuildProgram);
    MNN_LOAD_FUNCTION_PTR(clEnqueueNDRangeKernel);
    MNN_LOAD_FUNCTION_PTR(clSetKernelArg);
    MNN_LOAD_FUNCTION_PTR(clReleaseKernel);
    MNN_LOAD_FUNCTION_PTR(clCreateProgramWithSource);
    MNN_LOAD_FUNCTION_PTR(clCreateBuffer);
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
    MNN_LOAD_FUNCTION_PTR(clEnqueueCopyBuffer);
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
    MNN_LOAD_FUNCTION_PTR(clRetainEvent);
    MNN_LOAD_FUNCTION_PTR(clGetKernelWorkGroupInfo);
    MNN_LOAD_FUNCTION_PTR(clGetEventInfo);
    MNN_LOAD_FUNCTION_PTR(clGetEventProfilingInfo);
    MNN_LOAD_FUNCTION_PTR(clGetMemObjectInfo);
    MNN_LOAD_FUNCTION_PTR(clGetImageInfo);
    MNN_LOAD_FUNCTION_PTR(clEnqueueCopyImage);
    MNN_LOAD_FUNCTION_PTR(clEnqueueReadImage);
    MNN_LOAD_FUNCTION_PTR(clEnqueueWriteImage);
    
    MNN_LOAD_PROP_PTR(clCreateCommandQueueWithProperties);
    MNN_LOAD_SVM_PTR(clSVMAlloc);
    MNN_LOAD_SVM_PTR(clSVMFree);
    MNN_LOAD_SVM_PTR(clEnqueueSVMMap);
    MNN_LOAD_SVM_PTR(clEnqueueSVMUnmap);
    MNN_LOAD_SVM_PTR(clSetKernelArgSVMPointer);
#undef MNN_LOAD_FUNCTION_PTR

    return true;
}

OpenCLSymbolsOperator* OpenCLSymbolsOperator::createOpenCLSymbolsOperatorSingleInstance() {
    static std::once_flag sFlagInitSymbols;
    static OpenCLSymbolsOperator* gInstance = nullptr;
    std::call_once(sFlagInitSymbols, [&]() {
        gInstance = new OpenCLSymbolsOperator;
    });
    return gInstance;
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

cl_int CL_API_CALL clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetPlatformIDs;
    MNN_CHECK_NOTNULL(func);
    return func(num_entries, platforms, num_platforms);
}

cl_int CL_API_CALL clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size,
                         void *param_value, size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetPlatformInfo;
    MNN_CHECK_NOTNULL(func);
    return func(platform, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices,
                      cl_uint *num_devices) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetDeviceIDs;
    MNN_CHECK_NOTNULL(func);
    return func(platform, device_type, num_entries, devices, num_devices);
}

cl_int CL_API_CALL clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value,
                       size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetDeviceInfo;
    MNN_CHECK_NOTNULL(func);
    return func(device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_context CL_API_CALL clCreateContext(const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices,
                           void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *), void *user_data,
                           cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateContext;
    MNN_CHECK_NOTNULL(func);
    return func(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}

cl_context CL_API_CALL clCreateContextFromType(const cl_context_properties *properties, cl_device_type device_type,
                                   void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
                                   void *user_data, cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateContextFromType;
    MNN_CHECK_NOTNULL(func);
    return func(properties, device_type, pfn_notify, user_data, errcode_ret);
}

cl_int CL_API_CALL clRetainContext(cl_context context) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clRetainContext;
    MNN_CHECK_NOTNULL(func);
    return func(context);
}

cl_int CL_API_CALL clReleaseContext(cl_context context) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clReleaseContext;
    MNN_CHECK_NOTNULL(func);
    return func(context);
}

cl_int CL_API_CALL clGetContextInfo(cl_context context, cl_context_info param_name, size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetContextInfo;
    MNN_CHECK_NOTNULL(func);
    return func(context, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_program CL_API_CALL clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths,
                                     cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateProgramWithSource;
    MNN_CHECK_NOTNULL(func);
    return func(context, count, strings, lengths, errcode_ret);
}

cl_program CL_API_CALL clCreateProgramWithBinary(cl_context context, cl_uint count, const cl_device_id *           device_list, const size_t* length, const unsigned char ** buffer, cl_int * binary_status, cl_int * errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateProgramWithBinary;
    MNN_CHECK_NOTNULL(func);
    return func(context, count, device_list, length, buffer, binary_status, errcode_ret);
}

cl_int CL_API_CALL clGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetProgramInfo;
    MNN_CHECK_NOTNULL(func);
    return func(program, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int CL_API_CALL clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name,
                             size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetProgramBuildInfo;
    MNN_CHECK_NOTNULL(func);
    return func(program, device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int CL_API_CALL clRetainProgram(cl_program program) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clRetainProgram;
    MNN_CHECK_NOTNULL(func);
    return func(program);
}

cl_int CL_API_CALL clReleaseProgram(cl_program program) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clReleaseProgram;
    MNN_CHECK_NOTNULL(func);
    return func(program);
}

cl_int CL_API_CALL clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options,
                      void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data), void *user_data) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clBuildProgram;
    MNN_CHECK_NOTNULL(func);
    return func(program, num_devices, device_list, options, pfn_notify, user_data);
}

cl_kernel CL_API_CALL clCreateKernel(cl_program program, const char *kernelName, cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateKernel;
    MNN_CHECK_NOTNULL(func);
    return func(program, kernelName, errcode_ret);
}

cl_int CL_API_CALL clRetainKernel(cl_kernel kernel) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clRetainKernel;
    MNN_CHECK_NOTNULL(func);
    return func(kernel);
}

cl_int CL_API_CALL clReleaseKernel(cl_kernel kernel) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clReleaseKernel;
    MNN_CHECK_NOTNULL(func);
    return func(kernel);
}

cl_int CL_API_CALL clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clSetKernelArg;
    MNN_CHECK_NOTNULL(func);
    return func(kernel, arg_index, arg_size, arg_value);
}

cl_mem CL_API_CALL clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateBuffer;
    MNN_CHECK_NOTNULL(func);
    return func(context, flags, size, host_ptr, errcode_ret);
}

cl_int CL_API_CALL clRetainMemObject(cl_mem memobj) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clRetainMemObject;
    MNN_CHECK_NOTNULL(func);
    return func(memobj);
}

cl_int CL_API_CALL clReleaseMemObject(cl_mem memobj) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clReleaseMemObject;
    MNN_CHECK_NOTNULL(func);
    return func(memobj);
}

cl_int CL_API_CALL clGetImageInfo(cl_mem image, cl_image_info param_name, size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetImageInfo;
    MNN_CHECK_NOTNULL(func);
    return func(image, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int CL_API_CALL clRetainCommandQueue(cl_command_queue command_queue) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clRetainCommandQueue;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue);
}

cl_int CL_API_CALL clReleaseCommandQueue(cl_command_queue command_queue) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clReleaseCommandQueue;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue);
}

cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset,
                           size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                           cl_event *event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueReadBuffer;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list,
                event);
}

cl_int CL_API_CALL
clEnqueueCopyBuffer(cl_command_queue    command_queue,
                    cl_mem              src_buffer,
                    cl_mem              dst_buffer,
                    size_t              src_offset,
                    size_t              dst_offset,
                    size_t              size,
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueCopyBuffer;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, src_buffer, dst_buffer, src_offset, dst_offset, size, num_events_in_wait_list, event_wait_list, event);
}

cl_int CL_API_CALL clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset,
                            size_t size, const void *ptr, cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list, cl_event *event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueWriteBuffer;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list,
                event);
}

cl_int CL_API_CALL clEnqueueReadImage(cl_command_queue command_queue, cl_mem cl_image, cl_bool is_block, const size_t * origin, const size_t * region, size_t row_pitch,
                                       size_t slice_pitch, void * ptr, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event) CL_API_SUFFIX__VERSION_1_0 {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueReadImage;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, cl_image, is_block, origin, region, row_pitch, slice_pitch, ptr, num_events_in_wait_list, event_wait_list,
                event);
}

cl_int CL_API_CALL clEnqueueWriteImage(cl_command_queue command_queue, cl_mem cl_image, cl_bool is_block, const size_t * origin, const size_t * region, size_t row_pitch,
                                       size_t slice_pitch, const void * ptr, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueWriteImage;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, cl_image, is_block, origin, region, row_pitch, slice_pitch, ptr, num_events_in_wait_list, event_wait_list,
                event);
}

void* CL_API_CALL clEnqueueMapBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_map, cl_map_flags map_flags,
                         size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
                         cl_event *event, cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueMapBuffer;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list, event_wait_list,
                event, errcode_ret);
}

void* CL_API_CALL clEnqueueMapImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_map, cl_map_flags map_flags,
                        const size_t *origin, const size_t *region, size_t *image_row_pitch, size_t *image_slice_pitch,
                        cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event,
                        cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueMapImage;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, image, blocking_map, map_flags, origin, region, image_row_pitch, image_slice_pitch,
                num_events_in_wait_list, event_wait_list, event, errcode_ret);
}

cl_int CL_API_CALL clEnqueueUnmapMemObject(cl_command_queue command_queue, cl_mem memobj, void *mapped_ptr,
                               cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueUnmapMemObject;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
}

cl_int CL_API_CALL clGetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name,
                                size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetKernelWorkGroupInfo;
    MNN_CHECK_NOTNULL(func);
    return func(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int CL_API_CALL clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetEventProfilingInfo;
    MNN_CHECK_NOTNULL(func);
    return func(event, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int CL_API_CALL clGetMemObjectInfo(cl_mem memobj, cl_mem_info param_name, size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetMemObjectInfo;
    MNN_CHECK_NOTNULL(func);
    return func(memobj, param_name, param_value_size, param_value, param_value_size_ret);
}
cl_int CL_API_CALL clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
                              const size_t *global_work_offset, const size_t *global_work_size,
                              const size_t *local_work_size, cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list, cl_event *event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueNDRangeKernel;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size,
                num_events_in_wait_list, event_wait_list, event);
}

cl_int CL_API_CALL clWaitForEvents(cl_uint num_events, const cl_event *event_list) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clWaitForEvents;
    MNN_CHECK_NOTNULL(func);
    return func(num_events, event_list);
}

cl_int CL_API_CALL clRetainEvent(cl_event event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clRetainEvent;
    MNN_CHECK_NOTNULL(func);
    return func(event);
}

cl_int CL_API_CALL clReleaseEvent(cl_event event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clReleaseEvent;
    MNN_CHECK_NOTNULL(func);
    return func(event);
}

cl_int CL_API_CALL clGetEventInfo(cl_event event, cl_event_info param_name, size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clGetEventInfo;
    MNN_CHECK_NOTNULL(func);
    return func(event, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int CL_API_CALL clFlush(cl_command_queue command_queue) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clFlush;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue);
}

cl_int CL_API_CALL clFinish(cl_command_queue command_queue) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clFinish;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue);
}

cl_mem CL_API_CALL clCreateImage2D(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, size_t imageWidth,
                       size_t imageHeight, size_t image_row_pitch, void *host_ptr, cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateImage2D;
    MNN_CHECK_NOTNULL(func);
    return func(context, flags, image_format, imageWidth, imageHeight, image_row_pitch, host_ptr, errcode_ret);
}

cl_command_queue CL_API_CALL clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties,
                                      cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateCommandQueue;
    MNN_CHECK_NOTNULL(func);
    return func(context, device, properties, errcode_ret);
}
cl_int CL_API_CALL clEnqueueCopyImage(cl_command_queue queue,
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

// clCreateCommandQueueWithProperties wrapper
cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties(cl_context context, cl_device_id device, const cl_queue_properties *properties, cl_int *errcode_ret) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clCreateCommandQueueWithProperties;
    MNN_CHECK_NOTNULL(func);
    return func(context, device, properties, errcode_ret);
}

// clSVMAlloc wrapper, use OpenCLWrapper function.
void* CL_API_CALL clSVMAlloc(cl_context context, cl_mem_flags flags, size_t size, cl_uint align) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clSVMAlloc;
    MNN_CHECK_NOTNULL(func);
    return func(context, flags, size, align);
}

// clSVMFree wrapper, use OpenCLWrapper function.
void CL_API_CALL clSVMFree(cl_context context, void *buffer) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clSVMFree;
    MNN_CHECK_NOTNULL(func);
    func(context, buffer);
}

// clEnqueueSVMMap wrapper, use OpenCLWrapper function.
cl_int CL_API_CALL clEnqueueSVMMap(cl_command_queue command_queue, cl_bool blocking, cl_map_flags flags, void *host_ptr,
                       size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueSVMMap;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, blocking, flags, host_ptr, size, num_events_in_wait_list, event_wait_list, event);
}

// clEnqueueSVMUnmap wrapper, use OpenCLWrapper function.
cl_int CL_API_CALL clEnqueueSVMUnmap(cl_command_queue command_queue, void *host_ptr, cl_uint num_events_in_wait_list,
                         const cl_event *event_wait_list, cl_event *event) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clEnqueueSVMUnmap;
    MNN_CHECK_NOTNULL(func);
    return func(command_queue, host_ptr, num_events_in_wait_list, event_wait_list, event);
}

// clSetKernelArgSVMPointer wrapper, use OpenCLWrapper function.
cl_int CL_API_CALL clSetKernelArgSVMPointer(cl_kernel kernel, cl_uint index, const void *host_ptr) {
    auto func = MNN::OpenCLSymbolsOperator::getOpenclSymbolsPtr()->clSetKernelArgSVMPointer;
    MNN_CHECK_NOTNULL(func);
    return func(kernel, index, host_ptr);
}

#endif //MNN_USE_LIB_WRAPPER

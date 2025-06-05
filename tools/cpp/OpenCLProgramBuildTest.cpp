//
//  OpenCLProgramBuildTest.cpp
//  MNN
//
//  Created by MNN on 2025/5/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <fstream>
#include <string>
#include <vector>
#include "CL/cl.h"
#ifdef _WIN32
#include <windows.h>
#include <libloaderapi.h>
#else
#include <dlfcn.h>
#endif

using clGetPlatformIDsFunc = cl_int (CL_API_CALL *)(cl_uint, cl_platform_id *, cl_uint *);
using clBuildProgramFunc = cl_int (CL_API_CALL *)(cl_program, cl_uint, const cl_device_id *, const char *, void (CL_CALLBACK *pfn_notify)(cl_program, void *), void *);
using clCreateProgramWithSourceFunc = cl_program (CL_API_CALL *)(cl_context, cl_uint, const char **, const size_t *, cl_int *);
using clGetProgramBuildInfoFunc = cl_int (CL_API_CALL *)(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
using clCreateContextFunc = cl_context (CL_API_CALL *)(const cl_context_properties *, cl_uint, const cl_device_id *,
                                                       void(CL_CALLBACK *)( // NOLINT(readability/casting)
                                                           const char *, const void *, size_t, void *),
                                                       void *, cl_int *);
using clGetDeviceIDsFunc = cl_int (CL_API_CALL *)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
using clGetDeviceInfoFunc = cl_int (CL_API_CALL *)(cl_device_id, cl_device_info, size_t, void *, size_t *);
using clReleaseProgramFunc = cl_int (CL_API_CALL *)(cl_program program);
using clReleaseContextFunc = cl_int (CL_API_CALL *)(cl_context);
using clReleaseDeviceFunc = cl_int (CL_API_CALL *)(cl_device_id);


class OpenCLProgramTest {
public:
    OpenCLProgramTest(){
        static const std::vector<std::string> gOpencl_library_paths = {

        #if defined(__APPLE__) || defined(__MACOSX)
            "libOpenCL.so", "/System/Library/Frameworks/OpenCL.framework/OpenCL"
        #elif defined(__OHOS__)
            "/vendor/lib64/chipsetsdk/libhvgr_v200.so",
            "/vendor/lib64/chipsetsdk/libGLES_mali.so",
            "/system/lib64/libGLES_mali.so",
            "libGLES_mali.so",
            "/vendor/lib64/chipsetsdk/libEGI_imp1.so",
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
                mIsSupportAvailable = true;
            }
        }
        if(mIsSupportAvailable){
            cl_int err;
            err = clGetPlatformIDs(1, &platform, NULL);
            if (err != CL_SUCCESS) {
                printf("Failed to get platform ID err = %d\n", err);
                return ;
            }
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
            if (err != CL_SUCCESS) {
                printf("Failed to get device ID err = %d\n", err);
                return;
            }

            context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
            if (!context || err != CL_SUCCESS) {
                printf("Failed to create context err = %d\n", err);
                return;
            }
        }
    }
    bool LoadLibraryFromPath(const std::string &library_path){
        #if defined(_WIN32)
        handle_ = LoadLibraryA(library_path.c_str());
        if (handle_ == nullptr) {
            return false;
        }
        #define MNN_LOAD_FUNCTION_PTR(func_name) func_name = reinterpret_cast<func_name##Func>(GetProcAddress(static_cast<HMODULE>(handle_), #func_name));
        #else
        handle_ = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr) {
            return false;
        }
        #define MNN_LOAD_FUNCTION_PTR(func_name) func_name = reinterpret_cast<func_name##Func>(dlsym(handle_, #func_name));
        #endif
        MNN_LOAD_FUNCTION_PTR(clGetPlatformIDs);
        MNN_LOAD_FUNCTION_PTR(clBuildProgram);
        MNN_LOAD_FUNCTION_PTR(clCreateProgramWithSource);
        MNN_LOAD_FUNCTION_PTR(clGetProgramBuildInfo);
        MNN_LOAD_FUNCTION_PTR(clCreateContext);
        MNN_LOAD_FUNCTION_PTR(clGetDeviceIDs);
        MNN_LOAD_FUNCTION_PTR(clGetDeviceInfo);
        MNN_LOAD_FUNCTION_PTR(clReleaseProgram);
        MNN_LOAD_FUNCTION_PTR(clReleaseContext);
        MNN_LOAD_FUNCTION_PTR(clReleaseDevice);
        
        return true;
    }
    bool TestProgram(const std::vector<std::string> options){
        cl_int err;
        FILE* file = fopen("kernel.cl", "r");
        if (!file) {
            printf("Failed to open kernel file: kernel.cl\n");
            return false;
        }
        
        fseek(file, 0, SEEK_END);
        size_t fileSize = ftell(file);
        rewind(file);
        
        char* source = (char*)malloc(fileSize + 1);
        if (!source) {
            fclose(file);
            printf("Memory allocation failed for kernel source\n");
            return false;
        }
        
        fread(source, sizeof(char), fileSize, file);
        source[fileSize] = '\0';
        fclose(file);
        
        // test program
        const char *code = source;
        cl_program program = clCreateProgramWithSource(context, 1, &code, &fileSize, &err);
        if (!program || err != CL_SUCCESS) {
            printf("Failed to create program from source\n");
            return false;
        }
        for(int i = 0; i < options.size(); ++i){
            err = clBuildProgram(program, 1, &device, options[i].c_str(), NULL, NULL);
            if (err != CL_SUCCESS) {
                size_t logSize;
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
                char *buildLog = (char*)malloc(logSize);
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
                printf("Program build log: ");
                for (int i = 0; i < logSize; i++) {
                    printf("%c", buildLog[i]);
                }
                clReleaseProgram(program);
                free(buildLog);
                return false;
            }
        }
        
        clReleaseProgram(program);
        free(source);
        return true;
    }
    bool mIsSupportAvailable = false;
    ~OpenCLProgramTest(){
        if(mIsSupportAvailable){
            clReleaseDevice(device);
            clReleaseContext(context);
        }
        if (handle_ != nullptr) {
#if defined(_WIN32)
            FreeLibrary(static_cast<HMODULE>(handle_));
#else
            dlclose(handle_);
#endif
        }
    }
private:
    void *handle_ = nullptr;
    clGetPlatformIDsFunc clGetPlatformIDs;
    clBuildProgramFunc clBuildProgram;
    clCreateProgramWithSourceFunc clCreateProgramWithSource;
    clGetProgramBuildInfoFunc clGetProgramBuildInfo;
    clCreateContextFunc clCreateContext;
    clGetDeviceIDsFunc clGetDeviceIDs;
    clGetDeviceInfoFunc clGetDeviceInfo;
    clReleaseProgramFunc clReleaseProgram;
    clReleaseContextFunc clReleaseContext;
    clReleaseDeviceFunc clReleaseDevice;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
};

int main(int argc, char *argv[]) {
    std::string filename;
    if(argc > 1){
        filename = argv[1];
    }
    std::vector<std::string> options;
    std::fstream file("option.txt");
    if(file.is_open()){
        std::string line;
        while (getline(file, line)) { // 按行读取文件内容并输出
            options.push_back(line);
        }
        file.close();
    }
    printf("test filename is %s\n", filename.c_str());
    OpenCLProgramTest BuildTest;
    if(BuildTest.mIsSupportAvailable){
        if(BuildTest.TestProgram(options)){
            return 0;
        }
    }else{
        printf("OpenCL init fail\n");
        return -1;
    }
    return 0;
}


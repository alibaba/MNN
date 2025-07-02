#include <CL/cl.h>
#include <iostream>
#include <vector>

#define CHECK_ERR(err, msg) if (err != CL_SUCCESS) { std::cerr << msg << ": " << err << std::endl; exit(EXIT_FAILURE); }


int main() {
    const int SIZE = 10;
    std::vector<int> A(SIZE, 1);  // 初始化向量 A
    std::vector<int> B(SIZE, 2);  // 初始化向量 B
    std::vector<int> C(SIZE, 0);  // 结果向量 C

    cl_int err;
    cl_uint platformCount;
    cl_platform_id platform;
    cl_device_id device;

    // 获取平台和设备
    // 选择第1个可用的平台
    err = clGetPlatformIDs(1, &platform, &platformCount);
    CHECK_ERR(err, "Failed to get platform ID");
    // GPU类型的device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    CHECK_ERR(err, "Failed to get device ID");

    // 创建上下文
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_ERR(err, "Failed to create context");

    // 创建命令队列（使用 clCreateCommandQueue），用于跟底层设备通信，将要在设备上完成的操作都会在命令队列中排队。
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERR(err, "Failed to create command queue");

    // 分配和拷贝内存
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, A.data(), &err);
    CHECK_ERR(err, "Failed to create buffer A");

    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, B.data(), &err);
    CHECK_ERR(err, "Failed to create buffer B");

    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * SIZE, nullptr, &err);
    CHECK_ERR(err, "Failed to create buffer C");

    // 加载内核代码
    // 也可以从硬盘加载.cl文件，存储在一个字符串中，与下面等价。
    const char* kernelSource = R"(
        __kernel void vector_add(__global const int* A, __global const int* B, __global int* C) {
            int idx = get_global_id(0);
            C[idx] = A[idx] + B[idx];
        }
    )";
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    CHECK_ERR(err, "Failed to create program");

    // 编译程序
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    CHECK_ERR(err, "Failed to build program");

    // 创建内核
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    CHECK_ERR(err, "Failed to create kernel");

    // 设置内核参数
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

    // 设置工作项
    size_t globalWorkSize = SIZE;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    CHECK_ERR(err, "Failed to enqueue kernel");

    // 读取结果
    err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(int) * SIZE, C.data(), 0, nullptr, nullptr);
    CHECK_ERR(err, "Failed to read buffer");

    // 打印结果
    std::cout << "Result: ";
    for (int val : C) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 清理资源
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

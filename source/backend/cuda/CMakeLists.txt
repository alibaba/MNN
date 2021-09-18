set(CUDA_MIN_VERSION "9.0")
find_package(CUDA ${CUDA_MIN_VERSION})

if(CUDA_FOUND)
    include(${CMAKE_CURRENT_SOURCE_DIR}/SelectCudaComputeArch.cmake)
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${CUDA_ARCHS})
    # set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH_FLAGS}")
    # Fix for some combinations of CUDA and GCC (e.g. under Ubuntu 16.04).
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES -Wno-deprecated-gpu-targets -w")
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0")
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2")
    endif()
    if (WIN32)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler /FS")
    endif ()
    message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
                    " archs: ${CUDA_ARCH_FLAGS_readable})")
else()
    message(FATAL_ERROR "CUDA not found >= ${CUDA_MIN_VERSION} required)")
endif()

file(GLOB_RECURSE MNN_CUDA_SRC ${CMAKE_CURRENT_LIST_DIR}/core/* ${CMAKE_CURRENT_SOURCE_DIR}/execution/*)
message(STATUS "message ${CUDA_NVCC_FLAGS} !!!!!!!!!!!")

if(WIN32)
    cuda_add_library(MNN_CUDA STATIC Register.cpp ${MNN_CUDA_SRC})
    string(REPLACE "cublas.lib" "cudnn.lib" CUDNN_LIBRARIES ${CUDA_CUBLAS_LIBRARIES})
    set(MNN_CUDA_LIBS MNN_CUDA ${CUDNN_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES} PARENT_SCOPE)
else()
    cuda_add_library(MNN_Cuda_Main SHARED ${MNN_CUDA_SRC})
    set(MNN_CUDA_LIBS MNN_Cuda_Main cudnn cublas PARENT_SCOPE)
    add_library(MNN_CUDA OBJECT Register.cpp)
endif()

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include/
)


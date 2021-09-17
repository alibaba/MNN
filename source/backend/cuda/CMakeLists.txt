set(CUDA_MIN_VERSION "9.0")
find_package(CUDA ${CUDA_MIN_VERSION})

if(CUDA_FOUND)
    include(${CMAKE_CURRENT_SOURCE_DIR}/SelectCudaComputeArch.cmake)
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${CUDA_ARCHS})
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH_FLAGS}")
    # Fix for some combinations of CUDA and GCC (e.g. under Ubuntu 16.04).
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES -Wno-deprecated-gpu-targets -w -O2")
    message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
                    " archs: ${CUDA_ARCH_FLAGS_readable})")

    # first search cublas from CUDA_TOOLKIT_ROOT_DIR
    find_library(CUBLAS_LIBRARY cublas 
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64 
        ${CUDA_TOOLKIT_ROOT_DIR}/lib 
        NO_DEFAULT_PATH)

    # search default path if cannot find cublas in non-default
    find_library(CUBLAS_LIBRARY cublas)            

    # first search cudnn from CUDA_TOOLKIT_ROOT_DIR
    find_library(CUDNN_LIBRARY cudnn
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib
        NO_DEFAULT_PATH)
    # search default path if cannot find cudnn in non-default
    find_library(CUDNN_LIBRARY cudnn)    

else()
    message(FATAL_ERROR "CUDA not found >= ${CUDA_MIN_VERSION} required)")
endif()

file(GLOB_RECURSE MNN_CUDA_SRC ${CMAKE_CURRENT_LIST_DIR}/core/* ${CMAKE_CURRENT_SOURCE_DIR}/execution/*)
message(STATUS "message ${CUDA_NVCC_FLAGS} !!!!!!!!!!!")

# add_library(MNN_Cuda SHARED ${MNN_CUDA_SRC} )
cuda_add_library(MNN_Cuda_Main SHARED ${MNN_CUDA_SRC} )
set(MNN_CUDA_LIBS MNN_Cuda_Main ${CUDNN_LIBRARY} ${CUBLAS_LIBRARY} PARENT_SCOPE)
add_library(MNN_CUDA OBJECT Register.cpp)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include/
)



file(GLOB MNN_TRT_SRCS ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/backend/* ${CMAKE_CURRENT_LIST_DIR}/execution/*)
file(GLOB CUDA_SRCS ${CMAKE_CURRENT_LIST_DIR}/execution/plugin/*)

FIND_PACKAGE(CUDA REQUIRED)

add_library( MNN_TRT OBJECT ${MNN_TRT_SRCS})

include_directories(/usr/local/cuda/include/)
include_directories(${CMAKE_CURRENT_LIST_DIR}/backend/)
include_directories(${CMAKE_CURRENT_LIST_DIR}/execution/)
include_directories(${CMAKE_CURRENT_LIST_DIR}/execution/plugin/)
option(MNN_TRT_DYNAMIC "use dlopen to find trt" OFF)

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_70,code=sm_70
    -gencode arch=compute_75,code=sm_75
)

cuda_add_library(TRT_CUDA_PLUGIN SHARED ${CUDA_SRCS})
if (MNN_TRT_DYNAMIC)
    add_definitions(-DMNN_TRT_DYNAMIC)
    set(MNN_TRT_LIBS TRT_CUDA_PLUGIN ${CUDA_CUDA_LIBRARY} PARENT_SCOPE)
else()
    set(MNN_TRT_LIBS TRT_CUDA_PLUGIN ${CUDA_CUDA_LIBRARY} nvinfer PARENT_SCOPE)
endif()

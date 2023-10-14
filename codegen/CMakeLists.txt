option(MNN_CODEGEN_OPENCL "Build OpenCL op fuse." OFF)
option(MNN_CODEGEN_METAL "Build Metal op fuse." OFF)
option(MNN_CODEGEN_CUDA "Build Cuda op fuse." OFF)

file(GLOB MNN_FUSE_SRCS "${CMAKE_CURRENT_LIST_DIR}/*.*")

if(MNN_CODEGEN_OPENCL)
    add_definitions(-DMNN_CODEGEN_OPENCL)
    file(GLOB OPENCL_SRCS "${CMAKE_CURRENT_LIST_DIR}/opencl/*.*")
    list(APPEND MNN_FUSE_SRCS ${OPENCL_SRCS})
endif()

if(MNN_CODEGEN_METAL)
    add_definitions(-DMNN_CODEGEN_METAL)
    file(GLOB METAL_SRCS "${CMAKE_CURRENT_LIST_DIR}/metal/*.*")
    list(APPEND MNN_FUSE_SRCS ${METAL_SRCS})
endif()

if(MNN_CODEGEN_CUDA)
    add_definitions(-DMNN_CODEGEN_CUDA)
    file(GLOB CUDA_SRCS "${CMAKE_CURRENT_LIST_DIR}/cuda/*.*")
    list(APPEND MNN_FUSE_SRCS ${CUDA_SRCS})
endif()

add_library(MNNFuse OBJECT ${MNN_FUSE_SRCS})
# set_property(TARGET MNNFuse PROPERTY CXX_STANDARD 14)
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNFuse>)

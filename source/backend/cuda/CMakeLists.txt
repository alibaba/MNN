set(CUDA_MIN_VERSION "8.0")
find_package(CUDA ${CUDA_MIN_VERSION})

set (EXTRA_LIBS "")
if(MNN_CUDA_PROFILE)
    set(EXTRA_LIBS  -lnvToolsExt)
endif()

if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES -Wno-deprecated-gpu-targets -w ${EXTRA_LIBS}")
    if(MNN_SUPPORT_TRANSFORMER_FUSE)    
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++17") 
    endif()
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0")
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
    endif()
    if (WIN32)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler /FS")
    endif ()

    include(${CMAKE_CURRENT_SOURCE_DIR}/SelectCudaComputeArch.cmake)
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${CUDA_ARCHS})

    list(LENGTH CUDA_ARCH_FLAGS_readable_code arch_count)
    # Current Supported Arch List 
    IF (${arch_count} EQUAL 1)
        set(support_archs 60 61 62 70 72 75 80 86)
        list(FIND support_archs ${CUDA_ARCH_FLAGS_readable_code} list_index)
        IF (${list_index} EQUAL -1)
            message(FATAL_ERROR "Please add your own sm arch ${CUDA_ARCH_FLAGS_readable_code} to CmakeLists.txt!")
        ENDIF()
    ENDIF()

    IF ((CUDA_VERSION VERSION_GREATER "8.0") OR (CUDA_VERSION VERSION_EQUAL "8.0"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
    ENDIF()
    
    IF ((CUDA_VERSION VERSION_GREATER "10.1") OR (CUDA_VERSION VERSION_EQUAL "10.1"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_72,code=sm_72")
    ENDIF()

    IF ((CUDA_VERSION VERSION_GREATER "10.2") OR (CUDA_VERSION VERSION_EQUAL "10.2"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
        add_definitions(-DMNN_CUDA_ENABLE_SM75)
    ENDIF()

    IF ((CUDA_VERSION VERSION_GREATER "11.2") OR (CUDA_VERSION VERSION_EQUAL "11.2"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
        add_definitions(-DMNN_CUDA_ENABLE_SM80 -DMNN_CUDA_ENABLE_SM86)
    ENDIF()

    # Limit minimum cuda version for each archs

    IF (${arch_count} EQUAL 1)
        IF (MNN_SUPPORT_TRANSFORMER_FUSE OR (CUDA_ARCH_FLAGS_readable_code VERSION_GREATER "80") OR (CUDA_ARCH_FLAGS_readable_code VERSION_EQUAL "80"))
            IF (CUDA_VERSION VERSION_LESS "11.2")
                message(FATAL_ERROR "Please update cuda version to 11.2 or higher!")
            ENDIF()
        ENDIF()

        IF ((CUDA_ARCH_FLAGS_readable_code VERSION_GREATER "75") OR (CUDA_ARCH_FLAGS_readable_code VERSION_EQUAL "75"))
            IF (CUDA_VERSION VERSION_LESS "10.2")
                message(FATAL_ERROR "Please update cuda version to 10.2 or higher!")
            ENDIF()
        ENDIF()

        IF ((CUDA_ARCH_FLAGS_readable_code VERSION_GREATER "70") OR (CUDA_ARCH_FLAGS_readable_code VERSION_EQUAL "70"))
            IF (CUDA_VERSION VERSION_LESS "10.1")
                message(FATAL_ERROR "Please update cuda version to 10.1 or higher!")
            ENDIF()
        ENDIF()
    ENDIF()

    message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
                    " archs: ${CUDA_ARCH_FLAGS_readable})")
else()
    message(FATAL_ERROR "CUDA not found >= ${CUDA_MIN_VERSION} required)")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")

option(MNN_CUDA_QUANT "Enable MNN CUDA Quant File" OFF)
option(MNN_CUDA_BF16 "Enable MNN CUDA Bfloat16 File" OFF)
option(MNN_CUDA_TUNE_PARAM "Enable MNN CUDA Tuning Cutlass Params" OFF)

IF (MNN_CUDA_QUANT)
    add_definitions(-DENABLE_CUDA_QUANT)
ENDIF()

IF (MNN_CUDA_BF16)
    add_definitions(-DENABLE_CUDA_BF16)
ENDIF()

IF (MNN_CUDA_TUNE_PARAM)
    add_definitions(-DENABLE_CUDA_TUNE_PARAM)
ENDIF()

IF (MNN_LOW_MEMORY)
    add_definitions(-DMNN_LOW_MEMORY)
ENDIF()

file(GLOB_RECURSE MNN_CUDA_SRC ${CMAKE_CURRENT_LIST_DIR}/core/* ${CMAKE_CURRENT_SOURCE_DIR}/execution/*)
if(NOT MNN_SUPPORT_TRANSFORMER_FUSE)
    file(GLOB_RECURSE MNN_CUDA_TRANSFORMER_FUSE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/execution/plugin/*)
    list(REMOVE_ITEM MNN_CUDA_SRC ${MNN_CUDA_TRANSFORMER_FUSE_SRC})
endif()

if(NOT MNN_SUPPORT_RENDER)
    file(GLOB_RECURSE MNN_CUDA_RENDER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/execution/render/*)
    list(REMOVE_ITEM MNN_CUDA_SRC ${MNN_CUDA_RENDER_SRC})
endif()

if(NOT MNN_LOW_MEMORY)
    file(GLOB_RECURSE MNN_CUDA_LOW_MEM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/execution/weight_only_quant/*)
    list(REMOVE_ITEM MNN_CUDA_SRC ${MNN_CUDA_LOW_MEM_SRC})
endif()
message(STATUS "message ${CUDA_NVCC_FLAGS} !!!!!!!!!!! ${CUDA_INCLUDE_DIRS}")

if(WIN32)
    cuda_add_library(MNN_CUDA STATIC Register.cpp ${MNN_CUDA_SRC})
    set(MNN_CUDA_LIBS MNN_CUDA ${CUDA_LIBRARIES} PARENT_SCOPE)
else()

    cuda_add_library(MNN_Cuda_Main SHARED ${MNN_CUDA_SRC})
    if(MNN_CUDA_PROFILE)
      target_compile_options(MNN_Cuda_Main PRIVATE -DMNN_CUDA_PROFILE)
      target_link_libraries(MNN_Cuda_Main  ${CUDA_INCLUDE_DIRS}/../lib/libnvToolsExt.so)
    endif()

    if(MNN_CODEGEN_CUDA)
        target_link_libraries(MNN_Cuda_Main ${CUDA_INCLUDE_DIRS}/../lib/libnvrtc.so ${CUDA_INCLUDE_DIRS}/../lib/stubs/libcuda.so)
    endif()

    set(MNN_CUDA_LIBS MNN_Cuda_Main PARENT_SCOPE)
    add_library(MNN_CUDA OBJECT Register.cpp)
endif()

if(MNN_SUPPORT_TRANSFORMER_FUSE)
include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rd_party/cutlass/v3_4_0/include    
)
else()
include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rd_party/cutlass/v2_9_0/include
)
endif()



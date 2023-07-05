set(CUDA_MIN_VERSION "8.0")
find_package(CUDA ${CUDA_MIN_VERSION})

set (EXTRA_LIBS "")
if(MNN_CUDA_PROFILE)
    set(EXTRA_LIBS  -lnvToolsExt)
endif()


if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES -Wno-deprecated-gpu-targets -w ${EXTRA_LIBS}")
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
    ENDIF()

    IF ((CUDA_VERSION VERSION_GREATER "11.2") OR (CUDA_VERSION VERSION_EQUAL "11.2"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
    ENDIF()

    # Limit minimum cuda version for each archs
    IF (${arch_count} EQUAL 1)
        IF ((CUDA_ARCH_FLAGS_readable_code VERSION_GREATER "80") OR (CUDA_ARCH_FLAGS_readable_code VERSION_EQUAL "80"))
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

option(MNN_CUDA_QUANT "Enable MNN CUDA Quant File" OFF)
option(MNN_CUDA_BF16 "Enable MNN CUDA Bfloat16 File" OFF)

IF (MNN_CUDA_QUANT)
    add_definitions(-DENABLE_CUDA_QUANT)
ENDIF()

IF (MNN_CUDA_BF16)
    add_definitions(-DENABLE_CUDA_BF16)
ENDIF()

file(GLOB_RECURSE MNN_CUDA_SRC ${CMAKE_CURRENT_LIST_DIR}/core/* ${CMAKE_CURRENT_SOURCE_DIR}/execution/*)
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

    set(MNN_CUDA_LIBS MNN_Cuda_Main PARENT_SCOPE)
    add_library(MNN_CUDA OBJECT Register.cpp)
endif()

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rd_party/cutlass/include
)




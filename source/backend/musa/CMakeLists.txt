# MUSA Backend CMakeLists.txt
# 
# MUSA (Moore Threads GPU) Backend for MNN
# 
# This build script supports three modes:
# 1. Native MUSA SDK - Full MUSA support
# 2. CUDA compatibility - Map MUSA to CUDA (for testing/development)
# 3. Stub mode - Compile only, no GPU execution

# Include MUSA compatibility layer
include(${CMAKE_SOURCE_DIR}/3rd_party/musa_compat/CMakeLists.txt)

if(NOT MUSA_FOUND)
    message(WARNING "MUSA backend disabled: No MUSA/CUDA SDK found and stub mode not enabled")
    return()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")

option(MNN_MUSA_QUANT "Enable MNN MUSA Quant File" OFF)
option(MNN_MUSA_BF16 "Enable MNN MUSA Bfloat16 File" OFF)
option(MNN_MUSA_COMPAT_STUB "Use stub implementation (compile only)" ON)

IF (MNN_MUSA_QUANT)
    add_definitions(-DENABLE_MUSA_QUANT)
ENDIF()

IF (MNN_MUSA_BF16)
    add_definitions(-DENABLE_MUSA_BF16)
ENDIF()

IF (MNN_LOW_MEMORY)
    add_definitions(-DMNN_LOW_MEMORY)
ENDIF()

# Source files
file(GLOB_RECURSE MNN_MUSA_SRC ${CMAKE_CURRENT_LIST_DIR}/core/* ${CMAKE_CURRENT_SOURCE_DIR}/execution/*)

if(NOT MNN_SUPPORT_TRANSFORMER_FUSE)
    file(GLOB_RECURSE MNN_MUSA_TRANSFORMER_FUSE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/execution/plugin/*)
    list(REMOVE_ITEM MNN_MUSA_SRC ${MNN_MUSA_TRANSFORMER_FUSE_SRC})
endif()

# Include directories - use compat layer first
include_directories(
    ${CMAKE_SOURCE_DIR}/3rd_party/musa_compat/include
    ${MUSA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/
    ${CMAKE_SOURCE_DIR}/include/
)

# Build library based on available SDK
if(MNN_USE_NATIVE_MUSA)
    # Native MUSA build
    message(STATUS "Building MUSA backend with native MUSA SDK")
    if(WIN32)
        musa_add_library(MNN_MUSA STATIC Register.cpp ${MNN_MUSA_SRC})
        set(MNN_MUSA_LIBS MNN_MUSA ${MUSA_LIBRARIES} PARENT_SCOPE)
    else()
        musa_add_library(MNN_Musa_Main SHARED ${MNN_MUSA_SRC})
        set(MNN_MUSA_LIBS MNN_Musa_Main PARENT_SCOPE)
        add_library(MNN_MUSA OBJECT Register.cpp)
    endif()
    
elseif(MNN_USE_CUDA_AS_MUSA)
    # CUDA compatibility mode
    message(STATUS "Building MUSA backend with CUDA compatibility")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES -w")
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0")
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
    endif()
    
    # Rename .cu files to .cu for CUDA compilation
    foreach(SRC_FILE ${MNN_MUSA_SRC})
        if(SRC_FILE MATCHES "\\.cu$")
            list(APPEND MNN_MUSA_CU_SRC ${SRC_FILE})
        endif()
    endforeach()
    
    if(WIN32)
        cuda_add_library(MNN_MUSA STATIC Register.cpp ${MNN_MUSA_CU_SRC})
        set(MNN_MUSA_LIBS MNN_MUSA ${CUDA_LIBRARIES} PARENT_SCOPE)
    else()
        cuda_add_library(MNN_Musa_Main SHARED ${MNN_MUSA_CU_SRC})
        set(MNN_MUSA_LIBS MNN_Musa_Main PARENT_SCOPE)
        add_library(MNN_MUSA OBJECT Register.cpp)
    endif()
    
else()
    # Stub mode - compile C++ files only (skip .cu files)
    message(STATUS "Building MUSA backend in STUB mode (no GPU execution)")
    
    # Filter out .cu files, keep only .cpp/.hpp
    foreach(SRC_FILE ${MNN_MUSA_SRC})
        if(NOT SRC_FILE MATCHES "\\.cu$")
            list(APPEND MNN_MUSA_CPP_SRC ${SRC_FILE})
        endif()
    endforeach()
    
    add_library(MNN_MUSA OBJECT Register.cpp ${MNN_MUSA_CPP_SRC})
    set(MNN_MUSA_LIBS MNN_MUSA PARENT_SCOPE)
endif()

message(STATUS "MUSA Backend: Configured successfully")
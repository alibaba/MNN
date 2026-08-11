# MUSA Compatibility Layer CMake Configuration
# 
# This module provides MUSA API compatibility when the actual SDK is not available.
#
# Options:
#   MNN_MUSA_COMPAT_STUB    - Use stub implementation (compile only, no GPU)
#   MNN_MUSA_COMPAT_CUDA    - Map MUSA to CUDA (requires CUDA SDK)
#   MNN_MUSA_NATIVE         - Use native MUSA SDK (requires MUSA SDK)
#
# Priority: NATIVE > CUDA > STUB

cmake_minimum_required(VERSION 3.6)

# Check for native MUSA SDK first
if(MNN_MUSA_NATIVE AND NOT MNN_MUSA_COMPAT_STUB AND NOT MNN_MUSA_COMPAT_CUDA)
    find_package(MUSA QUIET)
    if(MUSA_FOUND)
        message(STATUS "MUSA Compat: Using native MUSA SDK")
        set(MNN_USE_NATIVE_MUSA ON)
        set(MUSA_COMPAT_INCLUDE_DIRS ${MUSA_INCLUDE_DIRS})
        set(MUSA_COMPAT_LIBRARIES ${MUSA_LIBRARIES})
    else()
        message(WARNING "MUSA SDK not found, falling back to compatibility layer")
        set(MUSA_FOUND FALSE)
    endif()
endif()

# Fallback to CUDA mapping
if(NOT MUSA_FOUND AND MNN_MUSA_COMPAT_CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "MUSA Compat: Mapping MUSA to CUDA")
        set(MNN_USE_CUDA_AS_MUSA ON)
        set(MUSA_COMPAT_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
        set(MUSA_COMPAT_LIBRARIES ${CUDA_LIBRARIES})
        set(MUSA_FOUND TRUE)
    else()
        message(WARNING "CUDA not found for MUSA compatibility")
    endif()
endif()

# Final fallback: stub implementation
if(NOT MUSA_FOUND OR MNN_MUSA_COMPAT_STUB)
    message(STATUS "MUSA Compat: Using stub implementation (compile only, no GPU)")
    set(MNN_USE_MUSA_STUB ON)
    set(MUSA_COMPAT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set(MUSA_COMPAT_LIBRARIES "")
    set(MUSA_FOUND TRUE)
endif()

# Export variables
set(MUSA_INCLUDE_DIRS ${MUSA_COMPAT_INCLUDE_DIRS} PARENT_SCOPE)
set(MUSA_LIBRARIES ${MUSA_COMPAT_LIBRARIES} PARENT_SCOPE)
set(MUSA_FOUND ${MUSA_FOUND} PARENT_SCOPE)

# Add compile definitions
if(MNN_USE_NATIVE_MUSA)
    add_definitions(-DMNN_USE_NATIVE_MUSA)
elseif(MNN_USE_CUDA_AS_MUSA)
    add_definitions(-DMNN_USE_CUDA_AS_MUSA)
elseif(MNN_USE_MUSA_STUB)
    add_definitions(-DMNN_USE_MUSA_STUB)
endif()

message(STATUS "MUSA Compat: Include dirs = ${MUSA_COMPAT_INCLUDE_DIRS}")
message(STATUS "MUSA Compat: Libraries = ${MUSA_COMPAT_LIBRARIES}")
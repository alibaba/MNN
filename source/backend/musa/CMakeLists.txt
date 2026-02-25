set(MUSA_MIN_VERSION "1.0")
find_package(MUSA ${MUSA_MIN_VERSION})

set(EXTRA_LIBS "")

if(MUSA_FOUND)
    set(MUSA_NVCC_FLAGS "${MUSA_NVCC_FLAGS} -D_FORCE_INLINES -w ${EXTRA_LIBS}")
    if(MNN_SUPPORT_TRANSFORMER_FUSE)    
        set(MUSA_NVCC_FLAGS "${MUSA_NVCC_FLAGS} --std=c++17") 
    endif()
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(MUSA_NVCC_FLAGS "${MUSA_NVCC_FLAGS} -O0")
    else()
        set(MUSA_NVCC_FLAGS "${MUSA_NVCC_FLAGS} -O3")
    endif()
    if (WIN32)
        set(MUSA_NVCC_FLAGS "${MUSA_NVCC_FLAGS} -Xcompiler /FS")
    endif ()

    message(STATUS "Enabling MUSA support (Moore Threads GPU)")
else()
    message(WARNING "MUSA not found, MUSA backend will not be built")
    return()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")

option(MNN_MUSA_QUANT "Enable MNN MUSA Quant File" OFF)
option(MNN_MUSA_BF16 "Enable MNN MUSA Bfloat16 File" OFF)

IF (MNN_MUSA_QUANT)
    add_definitions(-DENABLE_MUSA_QUANT)
ENDIF()

IF (MNN_MUSA_BF16)
    add_definitions(-DENABLE_MUSA_BF16)
ENDIF()

IF (MNN_LOW_MEMORY)
    add_definitions(-DMNN_LOW_MEMORY)
ENDIF()

file(GLOB_RECURSE MNN_MUSA_SRC ${CMAKE_CURRENT_LIST_DIR}/core/* ${CMAKE_CURRENT_SOURCE_DIR}/execution/*)

if(NOT MNN_SUPPORT_TRANSFORMER_FUSE)
    file(GLOB_RECURSE MNN_MUSA_TRANSFORMER_FUSE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/execution/plugin/*)
    list(REMOVE_ITEM MNN_MUSA_SRC ${MNN_MUSA_TRANSFORMER_FUSE_SRC})
endif()

message(STATUS "MUSA NVCC Flags: ${MUSA_NVCC_FLAGS}")

if(WIN32)
    musa_add_library(MNN_MUSA STATIC Register.cpp ${MNN_MUSA_SRC})
    set(MNN_MUSA_LIBS MNN_MUSA ${MUSA_LIBRARIES} PARENT_SCOPE)
else()
    musa_add_library(MNN_Musa_Main SHARED ${MNN_MUSA_SRC})
    set(MNN_MUSA_LIBS MNN_Musa_Main PARENT_SCOPE)
    add_library(MNN_MUSA OBJECT Register.cpp)
endif()

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/
    ${MUSA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include/
)

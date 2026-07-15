# source/backend/hexagon/CMakeLists.txt

if(NOT MNN_HEXAGON)
    return()
endif()

file(GLOB HEXAGON_BACKEND_SRCS ${CMAKE_CURRENT_LIST_DIR}/backend/*.cpp ${CMAKE_CURRENT_LIST_DIR}/backend/*.cc ${CMAKE_CURRENT_LIST_DIR}/backend/HexagonCommand.cpp)
file(GLOB HEXAGON_EXECUTION_SRCS ${CMAKE_CURRENT_LIST_DIR}/execution/*.cpp)

set(MNN_HEXAGON_SRCS
    ${HEXAGON_BACKEND_SRCS}
    ${HEXAGON_EXECUTION_SRCS}
)

add_library(MNN_Hexagon OBJECT ${MNN_HEXAGON_SRCS})

target_compile_definitions(MNN_Hexagon PRIVATE -DMNN_HEXAGON_ENABLED=1)
if (MNN_HEXAGON_ASAN)
    target_compile_definitions(MNN_Hexagon PRIVATE -DMNN_HEXAGON_ASAN=1)
endif()
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" OR ANDROID_ABI STREQUAL "arm64-v8a")
    set_source_files_properties(
        ${CMAKE_CURRENT_LIST_DIR}/backend/HexagonBackend.cpp
        PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16"
    )
endif()
if (MNN_GPU_TIME_PROFILE)
target_compile_definitions(MNN_Hexagon PRIVATE -DMNN_GPU_TIME_PROFILE)
endif()

target_include_directories(MNN_Hexagon
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/backend
        ${CMAKE_CURRENT_LIST_DIR}/execution
        ${CMAKE_CURRENT_LIST_DIR}/htp-ops-lib/include
        ${CMAKE_CURRENT_LIST_DIR}
        ${_HEXAGON_ROOT}/includes
)

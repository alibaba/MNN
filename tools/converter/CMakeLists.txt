cmake_minimum_required(VERSION 2.8)

project(MNNConverter)
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

set(PROJECT_VERSION "0.0.0.1")

configure_file(
    ./CONFIGURECONVERT.h.in
    ./CONFIGURECONVERT.h
)
include(cmake/macros.cmake)
# -----------debug or no-----------
option(MNN_CONVERT_DEBUG "Enable MNN CONVERT DEBUG" OFF)
option(MNN_BUILD_SHARED_LIBS "MNN build shared or static lib" ON)
if (WIN32)
    foreach(flag_var
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if (MNN_BUILD_SHARED_LIBS)
            if(${flag_var} MATCHES "/MT")
                string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
            endif()
        else ()
            if(${flag_var} MATCHES "/MD")
                string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
            endif()
        endif ()
    endforeach()
endif()

if(MNN_CONVERT_DEBUG)
    set(CMAKE_BUILD_TYPE "Debug")
    if(MSVC)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /DEBUG")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /DEBUG")
    else()
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    endif()
else()
    set(CMAKE_BUILD_TYPE "Release")
    add_definitions(-DNDEBUG)
endif()
if(SYSTEM.Linux)
    add_definitions(-fPIC)
endif()
option(TF_CONVERT_ORIGIN "Fall Back to Origin Model Converter" OFF)
if (TF_CONVERT_ORIGIN)
    add_definitions(-DTF_CONVERT_ORIGIN)
endif()

# -----------Enable tensorflow model optimizer or no-----------
option(TFMODEL_OPTIMIZE "Enable tensorflow model optimizer" OFF)
if(TFMODEL_OPTIMIZE)
    add_definitions(-DTFMODEL_OPTIMIZE)
endif()

# -----------find protobuf lib-----------
if (MSVC OR WIN32)
    set(Protobuf_SRC_ROOT_FOLDER $ENV{Protobuf_SRC_ROOT_FOLDER})
endif()
find_package(Protobuf REQUIRED)
if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
    set(Protobuf_LIBRARIES ${PROTOBUF_LIBRARIES})
    set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
endif()

# -----------set path-----------
set(SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(IR_PATH ${SRC_PATH}/../../../schema/current ${SRC_PATH}/../../../3rd_party/flatbuffers/include)
set(COMMON_PATH ${SRC_PATH}/common)
set(CAFFE_PATH ${SRC_PATH}/caffe)
set(TENSORFLOW_PATH ${SRC_PATH}/tensorflow)
set(ONNX_PATH ${SRC_PATH}/onnx)
set(TFLITE_PATH ${SRC_PATH}/tflite)
set(MNN_PATH ${SRC_PATH}/MNN)
set(OPTIMIZER_PATH ${SRC_PATH}/optimizer)
set(INCLUDE_PATH ${SRC_PATH}/../include ${SRC_PATH}/include)

# -----------set include path-----------
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${INCLUDE_PATH})
include_directories(${IR_PATH})
include_directories(${TFLITE_PATH}/schema)
include_directories(${SRC_PATH}/../../../include)
include_directories(${SRC_PATH}/../../../express/include)

add_subdirectory(${TENSORFLOW_PATH})
add_subdirectory(${ONNX_PATH})
add_subdirectory(${CAFFE_PATH})
add_subdirectory(${MNN_PATH})
add_subdirectory(${OPTIMIZER_PATH})
add_subdirectory(${TFLITE_PATH})

file(GLOB COMMON_SRC ${COMMON_PATH}/*)


add_library(COMMON_LIB STATIC ${COMMON_SRC})

add_executable(MNNDump2Json.out ${SRC_PATH}/MNNDump2Json.cpp)

set(COMMAND_SRC
    ${SRC_PATH}/cli.cpp
    ${SRC_PATH}/config.cpp
)

add_executable(MNNConvert ${SRC_PATH}/MNNConverter.cpp ${COMMAND_SRC})
target_include_directories(MNNConvert PRIVATE ${SRC_PATH})

if (MSVC OR WIN32)
    target_compile_options(MNNConvert PRIVATE "/wd4267")
    target_link_options(MNNConvert PRIVATE "/ignore:4099")
    foreach(DEPEND tensorflow caffe onnx tflite optimizer)
        target_link_options(MNNConvert PRIVATE /WHOLEARCHIVE:$<TARGET_FILE:${DEPEND}>)
    endforeach()
endif()

set(CONVERTER_DEPEND tensorflow caffe onnx mnn_bizcode tflite optimizer COMMON_LIB ${Protobuf_LIBRARIES})
if(NOT MNN_BUILD_SHARED_LIBS)
    if (BUILD_IOS OR APPLE)
        set(CONVERTER_DEPEND -Wl,-all_load ${CONVERTER_DEPEND} -Wl,-noall_load)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CONVERTER_DEPEND -Wl,--whole-archive ${CONVERTER_DEPEND} -Wl,--no-whole-archive)
    endif()
endif()
target_link_libraries(MNNConvert  ${CONVERTER_DEPEND}  ${END_EXTRA_LINK_FLAG})

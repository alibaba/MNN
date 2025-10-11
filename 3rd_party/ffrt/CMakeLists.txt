# -------------------------------------------------------------------
# ffrt CMake
#
# To enable thread sanitizer:
# -DCMAKE_CXX_FLAGS="-fsanitize=thread -g"
#
# To enable address and leak sanitizers
# -DCMAKE_CXX_FLAGS="-fsanitize=address  -fsanitize=leak -g"
# -------------------------------------------------------------------

# CMake version
cmake_minimum_required (VERSION 3.10)

# build options
option(FFRT_BBOX_ENABLE "Enables ffrt black box" ON)
option(FFRT_EXAMPLE "Enables ffrt examples compile" OFF)
option(FFRT_BENCHMARKS "Enables ffrt Benchmarks" OFF)
option(FFRT_TEST_ENABLE "Enables ffrt test" OFF)
option(FFRT_CLANG_COMPILE "use clang/clang++ for compiling" OFF)
option(FFRT_SANITIZE "enable address or thread sanitizer" OFF)

project(ffrt LANGUAGES C CXX ASM)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

#cmake build type settings,release or debug
if(CMAKE_BUILD_TYPE STREQUAL Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFFRT_RELEASE -DFFRT_LOG_LEVEL=0")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFFRT_LOG_LEVEL=3")
endif()

# project-specific variables
set(FFRT_CODE_PATH ${PROJECT_SOURCE_DIR}/src)
set(FFRT_INNER_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include)
set(FFRT_INNER_API_PATH ${PROJECT_SOURCE_DIR}/interfaces/inner_api)
set(FFRT_API_PATH ${PROJECT_SOURCE_DIR}/interfaces/kits)
set(FFRT_BUILD_PATH ${PROJECT_SOURCE_DIR}/build)
set(ROOT_PATH ${PROJECT_SOURCE_DIR}/../)
set(SECUREC_PATH ${PROJECT_SOURCE_DIR}/third_party/bounds_checking_function CACHE PATH
    "Path to securec")
set(TRACE_PATH ${FFRT_CODE_PATH}/dfx/trace)
set(QOS_PATH ${FFRT_CODE_PATH}/sched/)
set(securec_includes ${SECUREC_PATH}/include)
aux_source_directory(${SECUREC_PATH}/src securec_srcs)
add_library(securec SHARED ${securec_srcs})
target_include_directories(securec PRIVATE ${securec_includes})
include_directories(${securec_includes})

#compiler flags
if(FFRT_SANITIZE STREQUAL address)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer -DASAN_MODE")
elseif(FFRT_SANITIZE STREQUAL thread)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-omit-frame-pointer -DTSAN_MODE -DCLOSE_COROUTINE_MODE")
endif()

if(FFRT_UT_ENABLE STREQUAL OFF AND FFRT_ST_ENABLE STREQUAL OFF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
endif()
if(CMAKE_BUILD_TYPE STREQUAL Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O2 -fPIC")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O0 -fPIC")
endif()

# add the binary tree to the search path for include files
include_directories(${FFRT_API_PATH})
include_directories(${FFRT_INNER_API_PATH})
include_directories(${FFRT_INNER_INCLUDE_PATH})
include_directories(${FFRT_CODE_PATH})
include_directories(${TRACE_PATH})
include_directories(${QOS_PATH})

message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(STATUS "FFRT_BBOX_ENABLE: " ${FFRT_BBOX_ENABLE})
message(STATUS "FFRT_EXAMPLE: " ${FFRT_EXAMPLE})
message(STATUS "FFRT_BENCHMARKS: " ${FFRT_BENCHMARKS})
message(STATUS "FFRT_TEST_ENABLE: " ${FFRT_TEST_ENABLE})
message(STATUS "FFRT_CLANG_COMPILE: " ${FFRT_CLANG_COMPILE})
message(STATUS "FFRT_SANITIZE: " ${FFRT_SANITIZE})

add_definitions("-DFFRT_PTHREAD_ENABLE")
add_definitions("-DFFRT_ENABLE_TRAFFIC_MONITOR")

# ffrt qos
add_definitions(-DUSE_OHOS_QOS)

# ffrt BBOX
if(FFRT_BBOX_ENABLE STREQUAL ON)
  add_definitions("-DFFRT_BBOX_ENABLE")
endif()

# ffrt trace record level
add_definitions("-DFFRT_TRACE_RECORD_LEVEL=1")

# libffrt.so
add_subdirectory(src)
LINK_DIRECTORIES(${FFRT_BUILD_PATH}/src)

# ffrt Examples
if(FFRT_EXAMPLE STREQUAL ON)
  add_subdirectory(examples)
endif()

# ffrt Benchmarks
if(FFRT_BENCHMARKS STREQUAL ON)
  add_subdirectory(benchmarks)
endif()

# ffrt unit tests
if(FFRT_TEST_ENABLE STREQUAL ON)
  add_subdirectory(test)
endif()

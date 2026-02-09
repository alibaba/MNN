option(BUILD_MLS "Build PC Commandline." OFF)
option(LLM_USE_MINJA "Use minja to apply template" ON)
option(MNN_LLM_BUILD_DEMO "Build LLM demo" ON)
option(LLM_SUPPORT_HTTP_RESOURCE "Support HTTP resource download" ON)

set(LLM_DEPS ${MNN_DEPS})
if (MNN_BUILD_OPENCV)
    add_definitions(-DLLM_SUPPORT_VISION)
    if (MNN_SEP_BUILD)
        list(APPEND LLM_DEPS MNNOpenCV)
    endif()
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../tools/cv/include/)
endif()

if(MNN_IMGCODECS)
    add_definitions(-DMNN_IMGCODECS)
endif()

if (MNN_BUILD_AUDIO)
    if (MNN_SEP_BUILD)
        list(APPEND LLM_DEPS MNNAudio)
    endif()
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../tools/audio/include/)
endif()

# include dir
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

# source files
FILE(GLOB_RECURSE SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*)

if (MNN_SEP_BUILD)
    if (MNN_BUILD_SHARED_LIBS)
        # compile dynamic so, support Linux/Mac
        add_library(llm SHARED ${SRCS})
        set_target_properties(llm PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        target_link_libraries(llm ${LLM_DEPS})
    else()
        add_library(llm STATIC ${SRCS})
    endif()
    list(APPEND LLM_DEPS llm)
else()
    add_library(llm OBJECT ${SRCS})
endif()
if (LLM_USE_MINJA)
    target_compile_options(llm PRIVATE -DLLM_USE_MINJA)
    add_executable(apply_template ${CMAKE_CURRENT_LIST_DIR}/demo/apply_template.cpp)
    target_link_libraries(apply_template ${LLM_DEPS})
endif()

if (MNN_BUILD_OPENCV)
    target_compile_definitions(llm PRIVATE LLM_SUPPORT_VISION)
endif()
if (MNN_BUILD_AUDIO)
    add_definitions(-DLLM_SUPPORT_AUDIO)
endif()

if (LLM_SUPPORT_HTTP_RESOURCE)
    target_compile_definitions(llm PRIVATE LLM_SUPPORT_HTTP_RESOURCE)
    # Disable exceptions in httplib since MNN is compiled with -fno-exceptions
    target_compile_definitions(llm PRIVATE CPPHTTPLIB_NO_EXCEPTIONS)
endif()

IF(CMAKE_SYSTEM_NAME MATCHES "^Android" AND NOT MNN_BUILD_FOR_ANDROID_COMMAND)
IF(NOT NATIVE_INCLUDE_OUTPUT)
  set(NATIVE_INCLUDE_OUTPUT ".")
ENDIF()
add_custom_command(
  TARGET llm
  POST_BUILD
  COMMAND ${CMAKE_COMMAND}
  ARGS -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/include ${NATIVE_INCLUDE_OUTPUT}
)
ELSE()
INSTALL(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/ DESTINATION include FILES_MATCHING PATTERN *.hpp)
ENDIF()


if(MNN_LLM_BUILD_DEMO)
    add_executable(llm_demo ${CMAKE_CURRENT_LIST_DIR}/demo/llm_demo.cpp)
    add_executable(embedding_demo ${CMAKE_CURRENT_LIST_DIR}/demo/embedding_demo.cpp)
    add_executable(reranker_demo ${CMAKE_CURRENT_LIST_DIR}/demo/reranker_demo.cpp)
    add_executable(rollback_demo ${CMAKE_CURRENT_LIST_DIR}/demo/rollback_demo.cpp)
    add_executable(tokenizer_demo ${CMAKE_CURRENT_LIST_DIR}/demo/tokenizer_demo.cpp)

    include(${CMAKE_CURRENT_LIST_DIR}/tools/CMakeLists.txt)
    target_link_libraries(llm_demo ${LLM_DEPS})
    target_link_libraries(embedding_demo ${LLM_DEPS})
    target_link_libraries(reranker_demo ${LLM_DEPS})
    target_link_libraries(rollback_demo ${LLM_DEPS})
    target_link_libraries(tokenizer_demo ${LLM_DEPS})
endif()

if (BUILD_MLS)
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version" FORCE)

add_executable(mls
        ${CMAKE_CURRENT_LIST_DIR}/app/mls.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/remote_model_downloader.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/file_utils.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/llm_benchmark.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/llm_benchmark_common.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/hf_api_client.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/mls_server.cpp
)
# set(OPENSSL_ROOT_DIR "C:/Program Files/OpenSSL-Win64")
# set(OPENSSL_INCLUDE_DIR "C:/Program Files/OpenSSL-Win64/include")
# set(OPENSSL_CRYPTO_LIBRARY "C:/Program Files/OpenSSL-Win64/lib/VC/x64/MD/libcrypto.lib")
# set(OPENSSL_LIBRARIES "C:/Program Files/OpenSSL-Win64/lib//VC/x64/MD/libssl.lib;C:/Program Files/OpenSSL-Win64/lib/VC/x64/MD/libcrypto.lib")
# Set OpenSSL_FOUND based on the presence of include directory and libraries
set(OPENSSL_USE_STATIC_LIBS TRUE)
message(STATUS "OPENSSL_INCLUDE_DIR: ${OPENSSL_INCLUDE_DIR} OPENSSL_LIBRARIES:${OPENSSL_LIBRARIES}")
if(OPENSSL_LIBRARIES AND OPENSSL_INCLUDE_DIR)
    set(OpenSSL_FOUND TRUE)
else()
    set(OpenSSL_FOUND FALSE)
endif()
# Check if OPENSSL_LIBRARIES is not set to find OpenSSL
if(NOT OpenSSL_FOUND)
    find_package(OpenSSL REQUIRED)
endif()

# Check if OpenSSL is found
if(OpenSSL_FOUND)
    message(STATUS "OpenSSL found: ${OPENSSL_INCLUDE_DIR} ${OPENSSL_LIBRARIES}")

    # Use the OpenSSL include directory
    target_include_directories(mls PRIVATE ${OPENSSL_INCLUDE_DIR})

    # Link to OpenSSL libraries
    target_link_libraries(mls PRIVATE ${OPENSSL_LIBRARIES})

    # If on Apple platform, link with Security framework
    if(APPLE)
        find_library(SECURITY_FRAMEWORK Security)
        if(SECURITY_FRAMEWORK)
            target_link_libraries(mls PRIVATE ${SECURITY_FRAMEWORK})
        else()
            message(FATAL_ERROR "Could not find Security framework")
        endif()
    endif()
else()
    message(FATAL_ERROR "OpenSSL not found. Please install OpenSSL and set paths.")
endif()

target_include_directories(mls PRIVATE ${MNN_INCLUDES} ${CMAKE_CURRENT_LIST_DIR}/src/)
set_property(TARGET mls PROPERTY CXX_STANDARD 17)
set_property(TARGET mls PROPERTY CXX_STANDARD_REQUIRED ON)
# target_compile_options(mls PRIVATE -std=c++17)
target_link_libraries(mls PRIVATE ${LLM_DEPS})
target_compile_definitions(mls PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()

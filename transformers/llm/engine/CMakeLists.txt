option(LLM_SUPPORT_VISION "Llm model support vision input." ON)
option(LLM_SUPPORT_AUDIO "Llm model support audio input." ON)
option(BUILD_MLS "Build PC Commandline." OFF)


if (LLM_SUPPORT_VISION AND MNN_BUILD_OPENCV)
    include_directories(${CMAKE_SOURCE_DIR}/tools/cv/include/)
endif()

if (LLM_SUPPORT_AUDIO AND MNN_BUILD_AUDIO)
    list(APPEND MNN_DEPS MNNAudio)
    include_directories(${CMAKE_SOURCE_DIR}/tools/audio/include/)
endif()

# include dir
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

if (MNN_SEP_BUILD)
    if (MNN_BUILD_SHARED_LIBS)
        # compile dynamic so, support Linux/Mac
        add_library(llm SHARED ${SRCS})
        set_target_properties(llm PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        target_link_libraries(llm ${MNN_DEPS})
    else()
        add_library(llm STATIC ${SRCS})
    endif()
    list(APPEND MNN_DEPS llm)
else()
    add_library(llm OBJECT ${SRCS})
endif()

if (LLM_SUPPORT_VISION AND MNN_BUILD_OPENCV)
    target_compile_definitions(llm PRIVATE LLM_SUPPORT_VISION)
endif()
if (LLM_SUPPORT_AUDIO AND MNN_BUILD_AUDIO)
    target_compile_definitions(llm PRIVATE LLM_SUPPORT_AUDIO)
endif()

add_executable(llm_demo ${CMAKE_CURRENT_LIST_DIR}/llm_demo.cpp)
target_link_libraries(llm_demo ${MNN_DEPS})
add_executable(embedding_demo ${CMAKE_CURRENT_LIST_DIR}/embedding_demo.cpp)


target_link_libraries(embedding_demo ${MNN_DEPS})
add_executable(audio_demo ${CMAKE_CURRENT_LIST_DIR}/audio_demo.cpp)
target_link_libraries(audio_demo ${MNN_DEPS})
add_executable(rollback_demo ${CMAKE_CURRENT_LIST_DIR}/rollback_demo.cpp)
target_link_libraries(rollback_demo ${MNN_DEPS})

if (BUILD_MLS)
add_executable(mls
        ${CMAKE_CURRENT_LIST_DIR}/app/mls.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/remote_model_downloader.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/file_utils.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/llm_benchmark.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/hf_api_client.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/mls_server.cpp
)
# set(OPENSSL_ROOT_DIR "C:/Program Files/OpenSSL-Win64")
# set(OPENSSL_INCLUDE_DIR "C:/Program Files/OpenSSL-Win64/include")
# set(OPENSSL_CRYPTO_LIBRARY "C:/Program Files/OpenSSL-Win64/lib/VC/x64/MD/libcrypto.lib")
# set(OPENSSL_LIBRARIES "C:/Program Files/OpenSSL-Win64/lib//VC/x64/MD/libssl.lib;C:/Program Files/OpenSSL-Win64/lib/VC/x64/MD/libcrypto.lib")
# Set OpenSSL_FOUND based on the presence of include directory and libraries
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
    target_link_libraries(mls ${OPENSSL_LIBRARIES})

    # If on Apple platform, link with Security framework
    if(APPLE)
        find_library(SECURITY_FRAMEWORK Security)
        if(SECURITY_FRAMEWORK)
            target_link_libraries(mls ${SECURITY_FRAMEWORK})
        else()
            message(FATAL_ERROR "Could not find Security framework")
        endif()
    endif()
else()
    message(FATAL_ERROR "OpenSSL not found. Please install OpenSSL and set paths.")
endif()
set_property(TARGET mls PROPERTY CXX_STANDARD 17)
set_property(TARGET mls PROPERTY CXX_STANDARD_REQUIRED ON)
# target_compile_options(mls PRIVATE -std=c++17)
target_link_libraries(mls ${MNN_DEPS})
target_compile_definitions(mls PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()
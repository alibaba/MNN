cmake_minimum_required(VERSION 3.6)
project(mnncli VERSION 1.0.0 LANGUAGES CXX C ASM)

# C/C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version" FORCE)

option(BUILD_MNNCLI_TEST "Build MNNCLI test programs." OFF)

# Configure OpenSSL static libraries
if(APPLE)
    set(OPENSSL_ROOT_DIR "/opt/homebrew/opt/openssl@3")
    set(OPENSSL_USE_STATIC_LIBS TRUE)
    set(OPENSSL_INCLUDE_DIR "${OPENSSL_ROOT_DIR}/include" CACHE PATH "OpenSSL include directory" FORCE)
    set(OPENSSL_SSL_LIBRARY "${OPENSSL_ROOT_DIR}/lib/libssl.a" CACHE FILEPATH "OpenSSL SSL library" FORCE)
    set(OPENSSL_CRYPTO_LIBRARY "${OPENSSL_ROOT_DIR}/lib/libcrypto.a" CACHE FILEPATH "OpenSSL Crypto library" FORCE)
    set(OPENSSL_LIBRARIES "${OPENSSL_SSL_LIBRARY};${OPENSSL_CRYPTO_LIBRARY}" CACHE STRING "OpenSSL libraries" FORCE)
    set(OpenSSL_FOUND TRUE CACHE BOOL "OpenSSL found" FORCE)
    message(STATUS "Pre-configured OpenSSL static libraries:")
    message(STATUS "  Include: ${OPENSSL_INCLUDE_DIR}")
    message(STATUS "  SSL: ${OPENSSL_SSL_LIBRARY}")
    message(STATUS "  Crypto: ${OPENSSL_CRYPTO_LIBRARY}")
endif()

# Use httplib from MNN transformers directory instead of fetching it
# This ensures we use the same version (0.26.0) as the MNN library to avoid TLS mismatch

# Set MNN paths (can be overridden via command line)
set(MNN_BUILD_DIR "${CMAKE_CURRENT_LIST_DIR}/../../build_mnn_static" CACHE PATH "MNN build directory")
set(MNN_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH "MNN source directory")

# Find pre-built MNN library
find_library(MNN_LIBRARY 
    NAMES libMNN.a MNN
    PATHS ${MNN_BUILD_DIR}
    NO_DEFAULT_PATH
    REQUIRED
)

message(STATUS "Found MNN library: ${MNN_LIBRARY}")
message(STATUS "MNN source directory: ${MNN_SOURCE_DIR}")

# Add libyuv dependency from the local git submodule for Android builds
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # Ensure the submodule is available before adding it
    if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/../../3rd_party/libyuv/CMakeLists.txt)
        add_subdirectory(../../3rd_party/libyuv ${CMAKE_BINARY_DIR}/libyuv)
    else()
        message(FATAL_ERROR "libyuv submodule not found. Please ensure it is available in 3rd_party/libyuv.")
    endif()
endif()

# Function to convert file to C++ array
function(embed_file_as_c_array input_file output_file symbol_name)
    file(READ ${input_file} hex_content HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," hex_content ${hex_content})
    string(REGEX REPLACE ",$" "" hex_content ${hex_content})
    file(WRITE ${output_file} "const unsigned char ${symbol_name}[] = { ${hex_content} };\n")
endfunction()

# Generate embedded model_market.json data
set(MODEL_MARKET_JSON_FILE "${CMAKE_CURRENT_LIST_DIR}/../../apps/Android/MnnLlmChat/app/src/main/assets/model_market.json")
set(MODEL_MARKET_JSON_INC_FILE "${CMAKE_CURRENT_BINARY_DIR}/model_market_json_data.inc")

if(EXISTS ${MODEL_MARKET_JSON_FILE})
    embed_file_as_c_array(${MODEL_MARKET_JSON_FILE} ${MODEL_MARKET_JSON_INC_FILE} model_market_json_data)
else()
    message(WARNING "model_market.json not found at ${MODEL_MARKET_JSON_FILE}")
    # Create an empty file so the build doesn't fail
    file(WRITE ${MODEL_MARKET_JSON_INC_FILE} "const unsigned char model_market_json_data[] = { 0x7B, 0x7D }; // Empty JSON object {}\n")
endif()

set(MNNCLI_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/mnncli.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/model_runner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/log_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/model_file_downloader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/file_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/file_utils_config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/llm_benchmark.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../transformers/llm/engine/app/llm_benchmark_common.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/hf_api_client.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ms_api_client.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ms_model_downloader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ml_model_downloader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ml_api_client.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/mnncli_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/model_repo_downloader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/hf_file_metadata_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/hf_model_downloader.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/hf_sha_verifier.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/model_download_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/model_repository.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/model_market_data.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/model_name_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/model_sources.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/cli_download_listener.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/cli_config_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/llm_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/model_manager.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/local_model_utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/cli_command_handler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/cli_command_parser.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/cli_command_spec.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/handlers/run_command_handler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/handlers/delete_command_handler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/handlers/list_command_handler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/handlers/search_command_handler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/handlers/download_command_handler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/handlers/model_info_command_handler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/handlers/serve_command_handler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/handlers/benchmark_command_handler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/handlers/config_command_handler.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/handlers/info_command_handler.cpp
)

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    list(APPEND MNNCLI_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/AndroidVideoDecoder.cpp)
endif()

add_executable(mnncli ${MNNCLI_SOURCES})

# OpenCV configuration
if(MNN_BUILD_OPENCV)
    # Check if we're building for Android
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        message(STATUS "Android build detected - checking OpenCV configuration")
        
        # Check for external OpenCV Android SDK
        if(DEFINED OPENCV_ANDROID_SDK)
            message(STATUS "Using external OpenCV Android SDK: ${OPENCV_ANDROID_SDK}")
            
            # Set OpenCV paths for Android
            set(OpenCV_DIR "${OPENCV_ANDROID_SDK}/sdk/native/jni")
            set(OpenCV_INCLUDE_DIRS "${OPENCV_ANDROID_SDK}/sdk/native/jni/include")
            
            # Determine the correct ABI for OpenCV libraries
            if(DEFINED ANDROID_ABI)
                set(OpenCV_LIBS "${OPENCV_ANDROID_SDK}/sdk/native/libs/${ANDROID_ABI}/libopencv_java4.so")
            else()
                # Default to arm64-v8a if ABI not specified
                set(OpenCV_LIBS "${OPENCV_ANDROID_SDK}/sdk/native/libs/arm64-v8a/libopencv_java4.so")
            endif()
            
            # Verify OpenCV include directory exists
            if(EXISTS "${OpenCV_INCLUDE_DIRS}")
                message(STATUS "OpenCV include directory found: ${OpenCV_INCLUDE_DIRS}")
            else()
                message(FATAL_ERROR "OpenCV include directory not found: ${OpenCV_INCLUDE_DIRS}")
            endif()
            
            # Verify OpenCV library exists
            if(EXISTS "${OpenCV_LIBS}")
                message(STATUS "OpenCV library found: ${OpenCV_LIBS}")
            else()
                message(FATAL_ERROR "OpenCV library not found: ${OpenCV_LIBS}")
            endif()
            
            # Add OpenCV include directories
            target_include_directories(mnncli PRIVATE ${OpenCV_INCLUDE_DIRS})
            
            # Link OpenCV libraries
            target_link_libraries(mnncli PRIVATE ${OpenCV_LIBS})
            
            # Add OpenCV Android SDK path to include directories
            target_include_directories(mnncli PRIVATE 
                "${OPENCV_ANDROID_SDK}/sdk/native/jni/include"
                "${OPENCV_ANDROID_SDK}/sdk/native/jni/include/opencv4"
            )
            
            # Add compile definition to indicate OpenCV is available
            target_compile_definitions(mnncli PRIVATE OPENCV_AVAILABLE)
            
            set(ANDROID_EXTERNAL_OPENCV TRUE)
            message(STATUS "OpenCV Android SDK configured successfully")
        else()
            message(STATUS "No external OpenCV Android SDK found - using MNN's built-in implementation")
            # For Android, we'll use MNN's built-in OpenCV implementation
            # No need to find external OpenCV package
            set(ANDROID_OPENCV TRUE)
        endif()
    elseif(MNN_USE_SYSTEM_OPENCV)
        if(APPLE)
            set(OpenCV_DIR "/opt/homebrew/lib/cmake/opencv4")
        endif()
        find_package(OpenCV REQUIRED)
        if(OpenCV_FOUND)
            message(STATUS "OpenCV found: ${OpenCV_INCLUDE_DIRS} ${OpenCV_LIBS}")
            target_include_directories(mnncli PRIVATE ${OpenCV_INCLUDE_DIRS})
            target_link_libraries(mnncli PRIVATE ${OpenCV_LIBS})
            target_compile_definitions(mnncli PRIVATE OPENCV_AVAILABLE)
        endif()
    else()
        message(STATUS "Using MNN's built-in OpenCV implementation")
    endif()
endif()

# Set OpenSSL_FOUND based on the presence of include directory and libraries
set(OPENSSL_USE_STATIC_LIBS TRUE)
message(STATUS "OPENSSL_INCLUDE_DIR: ${OPENSSL_INCLUDE_DIR} OPENSSL_LIBRARIES:${OPENSSL_LIBRARIES}")

# Check if we're building for Android
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    message(STATUS "Android build detected - checking for OpenSSL Android build")
    
    # Link native media library and libyuv for Android
    message(STATUS "Linking Android native libraries: mediandk")
    target_link_libraries(mnncli PRIVATE mediandk)

    if(TARGET yuv)
        message(STATUS "Linking libyuv for Android.")
        target_link_libraries(mnncli PRIVATE yuv)
        # Explicitly add libyuv's include directory to fix 'libyuv.h' not found error
        target_include_directories(mnncli PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../3rd_party/libyuv/include)
    endif()

    # Add Android compile definition
    target_compile_definitions(mnncli PRIVATE ANDROID_BUILD)
    
    # Try to find OpenSSL Android build
    set(OPENSSL_ANDROID_CONFIG "${CMAKE_CURRENT_LIST_DIR}/../../3rd_party/openssl-android/openssl-android-config.cmake")
    if(EXISTS "${OPENSSL_ANDROID_CONFIG}")
        message(STATUS "Found OpenSSL Android config, including it")
        include("${OPENSSL_ANDROID_CONFIG}")
        set(OpenSSL_FOUND TRUE)
    else()
        message(STATUS "OpenSSL Android build not found, will be handled at runtime")
        set(OpenSSL_FOUND FALSE)
    endif()
    
    set(ANDROID_BUILD TRUE)
else()
    # For non-Android platforms, OpenSSL was already configured at the beginning
    if(NOT APPLE)
        # On other platforms, try to find OpenSSL
        set(OPENSSL_USE_STATIC_LIBS TRUE)
        find_package(OpenSSL REQUIRED)
        set(OpenSSL_FOUND TRUE)
    endif()
endif()

if(OpenSSL_FOUND)
    message(STATUS "OpenSSL found: ${OPENSSL_INCLUDE_DIR} ${OPENSSL_LIBRARIES}")

    # Use the OpenSSL include directory
    target_include_directories(mnncli PRIVATE ${OPENSSL_INCLUDE_DIR})

    # If on Apple platform, link with Security framework and Core Foundation first
    if(APPLE)
        find_library(SECURITY_FRAMEWORK Security)
        find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation)
        if(SECURITY_FRAMEWORK)
            target_link_libraries(mnncli PRIVATE ${SECURITY_FRAMEWORK})
        else()
            message(FATAL_ERROR "Could not find Security framework")
        endif()
        if(CORE_FOUNDATION_FRAMEWORK)
            target_link_libraries(mnncli PRIVATE ${CORE_FOUNDATION_FRAMEWORK})
        else()
            message(FATAL_ERROR "Could not find CoreFoundation framework")
        endif()
    endif()
else()
    if(ANDROID_BUILD)
        message(STATUS "OpenSSL not found - this is expected for Android builds")
    else()
        message(FATAL_ERROR "OpenSSL not found. Please install OpenSSL and set paths.")
    endif()
endif()

# Include directories
target_include_directories(mnncli PRIVATE
    ${MNN_SOURCE_DIR}/include/
    ${MNN_SOURCE_DIR}/transformers/llm/engine/src/
    ${MNN_SOURCE_DIR}/transformers/llm/engine/include/
    ${MNN_SOURCE_DIR}/3rd_party/
    ${CMAKE_CURRENT_LIST_DIR}/include/
    ${CMAKE_CURRENT_LIST_DIR}/../frameworks/3rd_party/include/
    ${CMAKE_CURRENT_BINARY_DIR}
)

set_property(TARGET mnncli PROPERTY CXX_STANDARD 17)
set_property(TARGET mnncli PROPERTY CXX_STANDARD_REQUIRED ON)

# Enable exceptions for mnncli (override global -fno-exceptions)
target_compile_options(mnncli PRIVATE -fexceptions)

# Enable RTTI for OpenCV compatibility
target_compile_options(mnncli PRIVATE -frtti)

# Suppress macOS version mismatch warnings for better compatibility
if(APPLE)
    target_compile_options(mnncli PRIVATE -Wno-deprecated-declarations)
    target_link_options(mnncli PRIVATE -Wl,-w)
endif()

# Link to pre-built MNN static library
target_link_libraries(mnncli PRIVATE ${MNN_LIBRARY})

# Disable Brotli support in httplib
target_compile_definitions(mnncli PRIVATE CPPHTTPLIB_NO_BROTLI)

# Link system libraries on macOS
if(APPLE)
    find_library(FOUNDATION Foundation REQUIRED)
    find_library(METAL Metal REQUIRED)
    find_library(GRAPHIC CoreGraphics REQUIRED)
    target_link_libraries(mnncli PRIVATE ${FOUNDATION} ${METAL} ${GRAPHIC})
endif()

# Link pthread and dl on Linux/macOS
if(UNIX)
    target_link_libraries(mnncli PRIVATE pthread dl)
endif()

# Link OpenSSL static libraries
if(OpenSSL_FOUND)
    # Add OpenSSL support
    target_compile_definitions(mnncli PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
    
    # Link OpenSSL static libraries - they were already configured as static at the beginning
    target_link_libraries(mnncli PRIVATE ${OPENSSL_LIBRARIES})
endif()

# Disable exceptions in httplib since MNN is compiled with -fno-exceptions
target_compile_definitions(mnncli PRIVATE CPPHTTPLIB_NO_EXCEPTIONS)

# Add test directory
add_subdirectory(test)

# Build test programs when BUILD_MNNCLI_TEST is enabled
if (BUILD_MNNCLI_TEST)
# Add test directory
endif()

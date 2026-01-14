cmake_minimum_required(VERSION 3.6)
project(mnncli VERSION 1.0.0 LANGUAGES CXX C ASM)

# C/C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version" FORCE)


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
else()
    find_package(OpenSSL REQUIRED)
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

# Add model downloader library
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../frameworks/model_downloader/cpp ${CMAKE_BINARY_DIR}/model_downloader)

set(MNNCLI_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/mnncli.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/model_runner.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/file_utils_config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/llm_benchmark.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../transformers/llm/engine/app/llm_benchmark_common.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/mnncli_server.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/model_repository.cpp
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

target_include_directories(mnncli PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${MNN_SOURCE_DIR}/include
    ${MNN_SOURCE_DIR}/transformers/llm/engine/include
    ${MNN_SOURCE_DIR}/transformers/llm/engine/src
    ${MNN_SOURCE_DIR}/3rd_party
    ${OPENSSL_INCLUDE_DIR}
    ${MNN_SOURCE_DIR}/transformers/llm/engine/app/jsonhpp
)

# ... OpenCV configuration ...

# Link to pre-built MNN static library
target_link_libraries(mnncli PRIVATE ${MNN_LIBRARY})

# Link model downloader
target_link_libraries(mnncli PRIVATE mnn_model_downloader)


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

# Build test programs when BUILD_MNNCLI_TEST is enabled
if (BUILD_MNNCLI_TEST)
    add_subdirectory(test)
endif()

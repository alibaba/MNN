cmake_minimum_required(VERSION 3.6)
project(mnn_model_downloader VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Function to convert file to C++ array
function(embed_file_as_c_array input_file output_file symbol_name)
    file(READ ${input_file} hex_content HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," hex_content ${hex_content})
    string(REGEX REPLACE ",$" "" hex_content ${hex_content})
    file(WRITE ${output_file} "const unsigned char ${symbol_name}[] = { ${hex_content} };\n")
endfunction()

# Generate embedded model_market.json data
set(MODEL_MARKET_JSON_FILE "${CMAKE_CURRENT_LIST_DIR}/../../../Android/MnnLlmChat/app/src/main/assets/model_market.json")
set(MODEL_MARKET_JSON_INC_FILE "${CMAKE_CURRENT_BINARY_DIR}/model_market_json_data.inc")

if(EXISTS ${MODEL_MARKET_JSON_FILE})
    embed_file_as_c_array(${MODEL_MARKET_JSON_FILE} ${MODEL_MARKET_JSON_INC_FILE} model_market_json_data)
else()
    message(WARNING "model_market.json not found at ${MODEL_MARKET_JSON_FILE}")
    # Create an empty file so the build doesn't fail
    file(WRITE ${MODEL_MARKET_JSON_INC_FILE} "const unsigned char model_market_json_data[] = { 0x7B, 0x7D }; // Empty JSON object {}\n")
endif()

# Source files
file(GLOB SOURCES 
    "src/*.cpp"
)

# Header files
file(GLOB HEADERS 
    "include/*.hpp"
)

# Create static library
add_library(mnn_model_downloader STATIC ${SOURCES})

# Include directories
target_include_directories(mnn_model_downloader PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)

# MNN Source Dir (Assumed to be set by parent or passed in)
if (NOT MNN_SOURCE_DIR)
    get_filename_component(MNN_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
endif()

target_include_directories(mnn_model_downloader PRIVATE
    ${MNN_SOURCE_DIR}/include/
    ${MNN_SOURCE_DIR}/transformers/llm/engine/src/
    ${MNN_SOURCE_DIR}/transformers/llm/engine/include/
    ${MNN_SOURCE_DIR}/3rd_party/
)

# OpenSSL Support (copied from mnncli)
if(APPLE)
    set(OPENSSL_ROOT_DIR "/opt/homebrew/opt/openssl@3")
    set(OPENSSL_USE_STATIC_LIBS TRUE)
    set(OPENSSL_INCLUDE_DIR "${OPENSSL_ROOT_DIR}/include")
    set(OPENSSL_LIBRARIES "${OPENSSL_ROOT_DIR}/lib/libssl.a" "${OPENSSL_ROOT_DIR}/lib/libcrypto.a")
    target_include_directories(mnn_model_downloader PRIVATE ${OPENSSL_INCLUDE_DIR})
    target_compile_definitions(mnn_model_downloader PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
    # Don't need to link here for static lib, but good to know
elseif(UNIX AND NOT ANDROID)
    find_package(OpenSSL REQUIRED)
    if(OpenSSL_FOUND)
        target_include_directories(mnn_model_downloader PRIVATE ${OPENSSL_INCLUDE_DIR})
        target_compile_definitions(mnn_model_downloader PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
    endif()
endif()

# Android OpenSSL
if(ANDROID)
    target_compile_definitions(mnn_model_downloader PRIVATE ANDROID_BUILD)
    # OpenSSL configuration usually propagated from parent or checking locally
    set(OPENSSL_ANDROID_CONFIG "${MNN_SOURCE_DIR}/3rd_party/openssl-android/openssl-android-config.cmake")
    if(EXISTS "${OPENSSL_ANDROID_CONFIG}")
        include("${OPENSSL_ANDROID_CONFIG}")
        if(OpenSSL_FOUND)
             target_include_directories(mnn_model_downloader PRIVATE ${OPENSSL_INCLUDE_DIR})
             target_compile_definitions(mnn_model_downloader PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
        endif()
    endif()
endif()

# Compiler options
target_compile_options(mnn_model_downloader PRIVATE -fexceptions)
target_compile_definitions(mnn_model_downloader PRIVATE CPPHTTPLIB_NO_BROTLI)
target_compile_definitions(mnn_model_downloader PRIVATE CPPHTTPLIB_NO_EXCEPTIONS)

# OpenCV if needed (check if verify logic uses it, likely not directly, but Image/Video utils might)
# Assuming simple downloader doesn't depend on OpenCV directly unless verified.

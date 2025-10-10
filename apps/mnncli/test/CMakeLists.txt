# Build MNNCLI test programs when BUILD_MNNCLI_TEST is enabled
if (BUILD_MNNCLI_TEST)
    # Core verification tests
    add_executable(hf_bin_file_verification_test
            src/hf_bin_file_verification_test.cpp
            ../src/hf_file_metadata_utils.cpp
            ../src/hf_api_client.cpp
            ../src/hf_sha_verifier.cpp
            ../src/model_file_downloader.cpp
            ../src/file_utils.cpp
            ../src/log_utils.cpp
    )

    add_executable(debug_sha_test
            src/debug_sha_test.cpp
            ../src/hf_sha_verifier.cpp
            ../src/log_utils.cpp
    )

    add_executable(verify_git_hash
            src/verify_git_hash.cpp
    )

    set_property(TARGET verify_git_hash PROPERTY CXX_STANDARD 17)
    set_property(TARGET verify_git_hash PROPERTY CXX_STANDARD_REQUIRED ON)

    add_executable(sha256_verification_test
            src/sha256_verification_test.cpp
            ../src/hf_sha_verifier.cpp
            ../src/log_utils.cpp
    )

    set_property(TARGET sha256_verification_test PROPERTY CXX_STANDARD 17)
    set_property(TARGET sha256_verification_test PROPERTY CXX_STANDARD_REQUIRED ON)

    target_include_directories(sha256_verification_test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../include/
        ${CMAKE_CURRENT_LIST_DIR}/../../frameworks/3rd_party/include/
    )

    add_executable(download_verification_test
            src/download_verification_test.cpp
            ../src/hf_file_metadata_utils.cpp
            ../src/hf_api_client.cpp
            ../src/hf_sha_verifier.cpp
            ../src/model_file_downloader.cpp
            ../src/file_utils.cpp
            ../src/log_utils.cpp
    )

    add_executable(embedding_file_verification_test
            src/embedding_file_verification_test.cpp
            ../src/hf_file_metadata_utils.cpp
            ../src/hf_api_client.cpp
            ../src/hf_sha_verifier.cpp
            ../src/model_file_downloader.cpp
            ../src/file_utils.cpp
            ../src/log_utils.cpp
    )

    add_executable(cdn_etag_comparison_test
            src/cdn_etag_comparison_test.cpp
            ../src/hf_file_metadata_utils.cpp
            ../src/hf_api_client.cpp
            ../src/hf_sha_verifier.cpp
            ../src/log_utils.cpp
            ../src/file_utils.cpp
    )

    add_executable(test_llm_weight
            src/test_llm_weight.cpp
            ../src/hf_file_metadata_utils.cpp
            ../src/hf_api_client.cpp
            ../src/hf_sha_verifier.cpp
            ../src/model_file_downloader.cpp
            ../src/file_utils.cpp
            ../src/log_utils.cpp
    )

    add_executable(test_real_file
            src/test_real_file.cpp
            ../src/hf_file_metadata_utils.cpp
            ../src/hf_api_client.cpp
            ../src/hf_sha_verifier.cpp
            ../src/model_file_downloader.cpp
            ../src/file_utils.cpp
            ../src/log_utils.cpp
    )

    set_property(TARGET download_verification_test PROPERTY CXX_STANDARD 17)
    set_property(TARGET download_verification_test PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET embedding_file_verification_test PROPERTY CXX_STANDARD 17)
    set_property(TARGET embedding_file_verification_test PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET cdn_etag_comparison_test PROPERTY CXX_STANDARD 17)
    set_property(TARGET cdn_etag_comparison_test PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET test_llm_weight PROPERTY CXX_STANDARD 17)
    set_property(TARGET test_llm_weight PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET test_real_file PROPERTY CXX_STANDARD 17)
    set_property(TARGET test_real_file PROPERTY CXX_STANDARD_REQUIRED ON)

    target_include_directories(download_verification_test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../include/
        ${CMAKE_CURRENT_LIST_DIR}/../../frameworks/3rd_party/include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../source/
        ${CMAKE_CURRENT_LIST_DIR}/../../../express/
        ${CMAKE_CURRENT_LIST_DIR}/../../../tools/
        ${CMAKE_CURRENT_LIST_DIR}/../../../codegen/
        ${CMAKE_CURRENT_LIST_DIR}/../../../schema/current/
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/flatbuffers/include
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/half
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/imageHelper
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/OpenCLHeaders/
        ${CMAKE_CURRENT_LIST_DIR}/../../frameworks/3rd_party/include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../transformers/llm/engine/src/
    )

    target_include_directories(embedding_file_verification_test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../include/
        ${CMAKE_CURRENT_LIST_DIR}/../../frameworks/3rd_party/include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../source/
        ${CMAKE_CURRENT_LIST_DIR}/../../../express/
        ${CMAKE_CURRENT_LIST_DIR}/../../../tools/
        ${CMAKE_CURRENT_LIST_DIR}/../../../codegen/
        ${CMAKE_CURRENT_LIST_DIR}/../../../schema/current/
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/flatbuffers/include
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/half
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/imageHelper
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/OpenCLHeaders/
        ${CMAKE_CURRENT_LIST_DIR}/../../frameworks/3rd_party/include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../transformers/llm/engine/src/
    )

    target_include_directories(cdn_etag_comparison_test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../include/
        ${CMAKE_CURRENT_LIST_DIR}/../../frameworks/3rd_party/include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../source/
        ${CMAKE_CURRENT_LIST_DIR}/../../../express/
        ${CMAKE_CURRENT_LIST_DIR}/../../../tools/
        ${CMAKE_CURRENT_LIST_DIR}/../../../codegen/
        ${CMAKE_CURRENT_LIST_DIR}/../../../schema/current/
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/flatbuffers/include
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/half
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/imageHelper
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/OpenCLHeaders/
        ${CMAKE_CURRENT_LIST_DIR}/../../frameworks/3rd_party/include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../transformers/llm/engine/src/
    )

    target_include_directories(test_llm_weight PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../include/
        ${CMAKE_CURRENT_LIST_DIR}/../../frameworks/3rd_party/include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../source/
        ${CMAKE_CURRENT_LIST_DIR}/../../../express/
        ${CMAKE_CURRENT_LIST_DIR}/../../../tools/
        ${CMAKE_CURRENT_LIST_DIR}/../../../codegen/
        ${CMAKE_CURRENT_LIST_DIR}/../../../schema/current/
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/flatbuffers/include
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/half
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/imageHelper
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/OpenCLHeaders/
        ${CMAKE_CURRENT_LIST_DIR}/../../frameworks/3rd_party/include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../transformers/llm/engine/src/
    )

    target_include_directories(test_real_file PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../include/
        ${CMAKE_CURRENT_LIST_DIR}/../../frameworks/3rd_party/include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../source/
        ${CMAKE_CURRENT_LIST_DIR}/../../../express/
        ${CMAKE_CURRENT_LIST_DIR}/../../../tools/
        ${CMAKE_CURRENT_LIST_DIR}/../../../codegen/
        ${CMAKE_CURRENT_LIST_DIR}/../../../schema/current/
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/flatbuffers/include
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/half
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/imageHelper
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/OpenCLHeaders/
        ${CMAKE_CURRENT_LIST_DIR}/../../frameworks/3rd_party/include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../transformers/llm/engine/src/
    )

    # Set OpenSSL_FOUND based on the presence of include directory and libraries
    set(OPENSSL_USE_STATIC_LIBS TRUE)

    # Check if we're building for Android
    if(CMAKE_SYSTEM_NAME STREQUAL "Android")
        message(STATUS "Android build detected - checking for OpenSSL Android build for tests")

        # Try to find OpenSSL Android build
        set(OPENSSL_ANDROID_CONFIG "${CMAKE_CURRENT_LIST_DIR}/../../3rd_party/openssl-android/openssl-android-config.cmake")
        if(EXISTS "${OPENSSL_ANDROID_CONFIG}")
            message(STATUS "Found OpenSSL Android config for tests, including it")
            include("${OPENSSL_ANDROID_CONFIG}")
            set(OpenSSL_FOUND TRUE)
        else()
            message(STATUS "OpenSSL Android build not found for tests, will be handled at runtime")
            set(OpenSSL_FOUND FALSE)
        endif()

        set(ANDROID_BUILD TRUE)
    else()
        # For non-Android platforms, check if OPENSSL_LIBRARIES is not set to find OpenSSL
        if(OPENSSL_LIBRARIES AND OPENSSL_INCLUDE_DIR)
            set(OpenSSL_FOUND TRUE)
        else()
            set(OpenSSL_FOUND FALSE)
        endif()

        if(NOT OpenSSL_FOUND)
            find_package(OpenSSL REQUIRED)
        endif()
    endif()

    # Check if OpenSSL is found
    if(OpenSSL_FOUND)
        message(STATUS "OpenSSL found: ${OPENSSL_INCLUDE_DIR} ${OPENSSL_LIBRARIES}")

        # Use the OpenSSL include directory
        target_include_directories(hf_bin_file_verification_test PRIVATE ${OPENSSL_INCLUDE_DIR})
        target_include_directories(debug_sha_test PRIVATE ${OPENSSL_INCLUDE_DIR})
        target_include_directories(verify_git_hash PRIVATE ${OPENSSL_INCLUDE_DIR})
        target_include_directories(sha256_verification_test PRIVATE ${OPENSSL_INCLUDE_DIR})
        target_include_directories(download_verification_test PRIVATE ${OPENSSL_INCLUDE_DIR})
        target_include_directories(embedding_file_verification_test PRIVATE ${OPENSSL_INCLUDE_DIR})
        target_include_directories(cdn_etag_comparison_test PRIVATE ${OPENSSL_INCLUDE_DIR})
        target_include_directories(test_llm_weight PRIVATE ${OPENSSL_INCLUDE_DIR})
        target_include_directories(test_real_file PRIVATE ${OPENSSL_INCLUDE_DIR})

        # Link to OpenSSL libraries
        target_link_libraries(hf_bin_file_verification_test PRIVATE ${OPENSSL_LIBRARIES})
        target_link_libraries(debug_sha_test PRIVATE ${OPENSSL_LIBRARIES})
        target_link_libraries(verify_git_hash PRIVATE ${OPENSSL_LIBRARIES})
        target_link_libraries(sha256_verification_test PRIVATE ${OPENSSL_LIBRARIES})
        target_link_libraries(download_verification_test PRIVATE ${OPENSSL_LIBRARIES})
        target_link_libraries(embedding_file_verification_test PRIVATE ${OPENSSL_LIBRARIES})
        target_link_libraries(cdn_etag_comparison_test PRIVATE ${OPENSSL_LIBRARIES})
        target_link_libraries(test_llm_weight PRIVATE ${OPENSSL_LIBRARIES})
        target_link_libraries(test_real_file PRIVATE ${OPENSSL_LIBRARIES})

        # If on Apple platform, link with Security framework and Core Foundation
        if(APPLE)
            find_library(SECURITY_FRAMEWORK Security)
            find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation)
            if(SECURITY_FRAMEWORK)
                target_link_libraries(hf_bin_file_verification_test PRIVATE ${SECURITY_FRAMEWORK})
                target_link_libraries(debug_sha_test PRIVATE ${SECURITY_FRAMEWORK})
                target_link_libraries(verify_git_hash PRIVATE ${SECURITY_FRAMEWORK})
                target_link_libraries(sha256_verification_test PRIVATE ${SECURITY_FRAMEWORK})
                target_link_libraries(download_verification_test PRIVATE ${SECURITY_FRAMEWORK})
                target_link_libraries(embedding_file_verification_test PRIVATE ${SECURITY_FRAMEWORK})
                target_link_libraries(cdn_etag_comparison_test PRIVATE ${SECURITY_FRAMEWORK})
                target_link_libraries(test_llm_weight PRIVATE ${SECURITY_FRAMEWORK})
                target_link_libraries(test_real_file PRIVATE ${SECURITY_FRAMEWORK})
            else()
                message(FATAL_ERROR "Could not find Security framework")
            endif()
            if(CORE_FOUNDATION_FRAMEWORK)
                target_link_libraries(hf_bin_file_verification_test PRIVATE ${CORE_FOUNDATION_FRAMEWORK})
                target_link_libraries(debug_sha_test PRIVATE ${CORE_FOUNDATION_FRAMEWORK})
                target_link_libraries(verify_git_hash PRIVATE ${CORE_FOUNDATION_FRAMEWORK})
                target_link_libraries(sha256_verification_test PRIVATE ${CORE_FOUNDATION_FRAMEWORK})
                target_link_libraries(download_verification_test PRIVATE ${CORE_FOUNDATION_FRAMEWORK})
                target_link_libraries(embedding_file_verification_test PRIVATE ${CORE_FOUNDATION_FRAMEWORK})
                target_link_libraries(cdn_etag_comparison_test PRIVATE ${CORE_FOUNDATION_FRAMEWORK})
                target_link_libraries(test_llm_weight PRIVATE ${CORE_FOUNDATION_FRAMEWORK})
                target_link_libraries(test_real_file PRIVATE ${CORE_FOUNDATION_FRAMEWORK})
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

    # Use MNN project's standard include directories and dependencies
    target_include_directories(hf_bin_file_verification_test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../../../include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../source/
        ${CMAKE_CURRENT_LIST_DIR}/../../../express/
        ${CMAKE_CURRENT_LIST_DIR}/../../../tools/
        ${CMAKE_CURRENT_LIST_DIR}/../../../codegen/
        ${CMAKE_CURRENT_LIST_DIR}/../../../schema/current/
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/flatbuffers/include
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/half
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/imageHelper
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/OpenCLHeaders/
        ${CMAKE_CURRENT_LIST_DIR}/../include/
        ${CMAKE_CURRENT_LIST_DIR}/../../frameworks/3rd_party/include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../transformers/llm/engine/src/
    )

    target_include_directories(debug_sha_test PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../../../include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../source/
        ${CMAKE_CURRENT_LIST_DIR}/../../../express/
        ${CMAKE_CURRENT_LIST_DIR}/../../../tools/
        ${CMAKE_CURRENT_LIST_DIR}/../../../codegen/
        ${CMAKE_CURRENT_LIST_DIR}/../../../schema/current/
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/flatbuffers/include
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/half
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/imageHelper
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/OpenCLHeaders/
        ${CMAKE_CURRENT_LIST_DIR}/../include/
        ${CMAKE_CURRENT_LIST_DIR}/../../frameworks/3rd_party/include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../transformers/llm/engine/src/
    )

    # Add system include directories for standard C++ headers
    target_include_directories(hf_bin_file_verification_test SYSTEM PRIVATE
        /usr/local/include
        /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/v1
        /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/17/include
        /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include
        /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include
    )

    target_include_directories(debug_sha_test SYSTEM PRIVATE
        /usr/local/include
        /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/v1
        /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/17/include
        /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include
        /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include
    )

    set_property(TARGET hf_bin_file_verification_test PROPERTY CXX_STANDARD 17)
    set_property(TARGET hf_bin_file_verification_test PROPERTY CXX_STANDARD_REQUIRED ON)
    set_property(TARGET debug_sha_test PROPERTY CXX_STANDARD 17)
    set_property(TARGET debug_sha_test PROPERTY CXX_STANDARD_REQUIRED ON)

    # Enable exceptions for test (override global -fno-exceptions)
    target_compile_options(hf_bin_file_verification_test PRIVATE -fexceptions)
    target_compile_options(debug_sha_test PRIVATE -fexceptions)
    target_compile_options(cdn_etag_comparison_test PRIVATE -fexceptions)

    # Enable standard C++ headers
    target_compile_options(hf_bin_file_verification_test PRIVATE -std=c++17)
    target_compile_options(debug_sha_test PRIVATE -std=c++17)
    target_compile_options(cdn_etag_comparison_test PRIVATE -std=c++17)
    target_compile_options(test_llm_weight PRIVATE -std=c++17)
    target_compile_options(test_real_file PRIVATE -std=c++17)

    # Link to required libraries
    target_link_libraries(hf_bin_file_verification_test PRIVATE ${MNN_DEPS})
    target_link_libraries(debug_sha_test PRIVATE ${MNN_DEPS})
    target_link_libraries(test_llm_weight PRIVATE ${MNN_DEPS})
    target_link_libraries(test_real_file PRIVATE ${MNN_DEPS})

    # Only add OpenSSL support compile definition when OpenSSL is available
    if(OpenSSL_FOUND)
        target_compile_definitions(hf_bin_file_verification_test PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
        target_compile_definitions(debug_sha_test PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
        target_compile_definitions(sha256_verification_test PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
        target_compile_definitions(download_verification_test PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
        target_compile_definitions(embedding_file_verification_test PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
        target_compile_definitions(cdn_etag_comparison_test PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
        target_compile_definitions(test_llm_weight PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
        target_compile_definitions(test_real_file PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
    endif()

endif()
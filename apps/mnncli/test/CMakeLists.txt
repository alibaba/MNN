# Build MNNCLI test programs when BUILD_MNNCLI is enabled
if (BUILD_MNNCLI)
    add_executable(ms_model_downloader_test
            ms_model_downloader_test.cpp
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
        target_include_directories(ms_model_downloader_test PRIVATE ${OPENSSL_INCLUDE_DIR})

        # Link to OpenSSL libraries
        target_link_libraries(ms_model_downloader_test PRIVATE ${OPENSSL_LIBRARIES})

        # If on Apple platform, link with Security framework and Core Foundation
        if(APPLE)
            find_library(SECURITY_FRAMEWORK Security)
            find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation)
            if(SECURITY_FRAMEWORK)
                target_link_libraries(ms_model_downloader_test PRIVATE ${SECURITY_FRAMEWORK})
            else()
                message(FATAL_ERROR "Could not find Security framework")
            endif()
            if(CORE_FOUNDATION_FRAMEWORK)
                target_link_libraries(ms_model_downloader_test PRIVATE ${CORE_FOUNDATION_FRAMEWORK})
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
    target_include_directories(ms_model_downloader_test PRIVATE 
        ${MNN_INCLUDES} 
        ${CMAKE_CURRENT_LIST_DIR}/../include/ 
        ${CMAKE_CURRENT_LIST_DIR}/../../../include/ 
        ${CMAKE_CURRENT_LIST_DIR}/../../../3rd_party/
        ${CMAKE_CURRENT_LIST_DIR}/../../frameworks/3rd_party/include/
        ${CMAKE_CURRENT_LIST_DIR}/../../../transformers/llm/engine/src/
    )
    
    set_property(TARGET ms_model_downloader_test PROPERTY CXX_STANDARD 17)
    set_property(TARGET ms_model_downloader_test PROPERTY CXX_STANDARD_REQUIRED ON)
    
    # Enable exceptions for test (override global -fno-exceptions)
    target_compile_options(ms_model_downloader_test PRIVATE -fexceptions)
    
    # Link to required libraries
    target_link_libraries(ms_model_downloader_test PRIVATE ${MNN_DEPS})
    
    # Only add OpenSSL support compile definition when OpenSSL is available
    if(OpenSSL_FOUND)
        target_compile_definitions(ms_model_downloader_test PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
    endif()
    

endif()

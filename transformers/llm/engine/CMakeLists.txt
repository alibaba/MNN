option(LLM_SUPPORT_VISION "Llm model support vision input." ON)
option(LLM_SUPPORT_AUDIO "Llm model support audio input." ON)


if (LLM_SUPPORT_VISION AND MNN_BUILD_OPENCV)
    list(APPEND MNN_DEPS MNNOpenCV)
    include_directories(${CMAKE_SOURCE_DIR}/tools/cv/include/)
endif()

if (LLM_SUPPORT_AUDIO AND MNN_BUILD_AUDIO)
    list(APPEND MNN_DEPS MNNAudio)
    include_directories(${CMAKE_SOURCE_DIR}/tools/audio/include/)
endif()

# include dir
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)
MESSAGE(STATUS "include dir: ${CMAKE_CURRENT_LIST_DIR}/include/")

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

#add_definitions(-DCPPHTTPLIB_OPENSSL_SUPPORT)

add_executable(mls
        ${CMAKE_CURRENT_LIST_DIR}/app/mls.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/remote_model_downloader.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/file_utils.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/llm_benchmark.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/hf_api_client.cpp
        ${CMAKE_CURRENT_LIST_DIR}/app/mls_server.cpp
)

target_compile_options(mls PRIVATE -std=c++17)

# Find OpenSSL
find_package(OpenSSL REQUIRED)

if (OpenSSL_FOUND)
    message(STATUS "OpenSSL found: ${OPENSSL_INCLUDE_DIR} ${OPENSSL_LIBRARIES}")
    target_include_directories(mls PRIVATE ${OPENSSL_INCLUDE_DIR})
    target_link_libraries(mls ${OPENSSL_LIBRARIES})
    if(APPLE)
        find_library(SECURITY_FRAMEWORK Security)
        if(SECURITY_FRAMEWORK)
            target_link_libraries(mls  ${SECURITY_FRAMEWORK})
        else()
            message(FATAL_ERROR "Could not find Security framework")
        endif()
    endif()
else()
    message(FATAL_ERROR "OpenSSL not found. Please install OpenSSL and set paths.")
endif()

target_link_libraries(mls ${MNN_DEPS} llm)
target_compile_definitions(mls PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
target_link_libraries(embedding_demo ${MNN_DEPS})
add_executable(audio_demo ${CMAKE_CURRENT_LIST_DIR}/audio_demo.cpp)
target_link_libraries(audio_demo ${MNN_DEPS})
add_executable(rollback_demo ${CMAKE_CURRENT_LIST_DIR}/rollback_demo.cpp)
target_link_libraries(rollback_demo ${MNN_DEPS})

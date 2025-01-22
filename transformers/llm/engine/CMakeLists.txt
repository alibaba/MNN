option(LLM_SUPPORT_VISION "Llm model support vision input." OFF)
option(LLM_SUPPORT_AUDIO "Llm model support audio input." OFF)


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

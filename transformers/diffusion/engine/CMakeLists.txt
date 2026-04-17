cmake_minimum_required(VERSION 3.10)

# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
include_directories(${CMAKE_SOURCE_DIR}/tools/cv/include/)

include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)
if (MNN_BUILD_LLM)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/../../llm/engine/src/)
endif()

if (MNN_SEP_BUILD)
    if (MNN_BUILD_SHARED_LIBS)
        # compile dynamic so, support Linux/Mac
        add_library(diffusion SHARED ${SRCS})
        set_target_properties(diffusion PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
        target_link_libraries(diffusion ${MNN_DEPS})
    else()
        add_library(diffusion STATIC ${SRCS})
    endif()
    list(APPEND MNN_DEPS diffusion)
else()
    add_library(diffusion OBJECT ${SRCS})
endif()

if (MNN_BUILD_LLM)
    target_compile_definitions(diffusion PRIVATE MNN_DIFFUSION_WITH_LLM_TOKENIZER)
endif()

add_executable(diffusion_demo 
    ${CMAKE_CURRENT_LIST_DIR}/diffusion_demo.cpp
)
target_link_libraries(diffusion_demo  ${MNN_DEPS})

if (MNN_BUILD_LLM)
    # sana_diffusion_demo depends on SanaLlm, which includes llm/llm.hpp and links against the llm engine.
    add_executable(sana_diffusion_demo ${CMAKE_CURRENT_LIST_DIR}/sana_diffusion_demo.cpp)
    target_link_libraries(sana_diffusion_demo ${MNN_DEPS})
endif()

add_executable(diffusion_sd35_demo ${CMAKE_CURRENT_LIST_DIR}/diffusion_sd35_demo.cpp)
target_link_libraries(diffusion_sd35_demo  ${MNN_DEPS})

if(MNN_NPU_BACKENDS_SHARED AND
   CMAKE_SYSTEM_NAME MATCHES "^Android" AND
   NOT CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
        message(STATUS "HiAI backend plugin is available only for Android arm64-v8a")
        return()
endif()

file(GLOB_RECURSE MNN_NPU_SRCS ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
list(FILTER MNN_NPU_SRCS EXCLUDE REGEX "/HiAIBackendPlugin\\.cpp$")
list(APPEND MNN_NPU_SRCS
        ${CMAKE_CURRENT_LIST_DIR}/backend/HiAIDynamicTrampolines.S)

IF(MNN_NPU_BACKENDS_SHARED)
        add_library(
                MNN_NPU
                SHARED
                ${MNN_NPU_SRCS}
                ${CMAKE_CURRENT_LIST_DIR}/HiAIBackendPlugin.cpp
        )
        set_target_properties(MNN_NPU PROPERTIES
                              OUTPUT_NAME "MNN_Backend_HiAI")
        target_compile_definitions(MNN_NPU PRIVATE
                MNN_HIAI_EXPLICIT_PLUGIN)
        target_link_libraries(MNN_NPU PRIVATE
                MNN
                ${CMAKE_DL_LIBS}
        )
ELSEIF(MNN_SEP_BUILD)
        add_library(
                MNN_NPU 
                SHARED 
                ${MNN_NPU_SRCS}
        )
        target_link_libraries(MNN_NPU PRIVATE
                MNN
                ${CMAKE_DL_LIBS}
        )
ELSE()
        add_library(
                MNN_NPU 
                OBJECT 
                ${MNN_NPU_SRCS}
        )
ENDIF()
target_include_directories(MNN_NPU PRIVATE ${CMAKE_CURRENT_LIST_DIR}/backend/)
target_include_directories(MNN_NPU PRIVATE ${CMAKE_CURRENT_LIST_DIR}/3rdParty/include/)
target_include_directories(MNN_NPU PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include/)

set(MNN_NPU_PUB_HDRS
    ${MNN_NPU_PUB_HDRS}
    "${CMAKE_CURRENT_LIST_DIR}/include/MNNHiAIBackend.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/MNNHiAIIO.h"
    "${CMAKE_CURRENT_LIST_DIR}/include/MNNHiAIPlugin.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/MNNHiAISession.hpp"
    PARENT_SCOPE)

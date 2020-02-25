FILE(GLOB_RECURSE MNN_Vulkan_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
option(MNN_VULKAN_REGEN "Regenerate Vulkan Shader binaries. Requires FULL glslang suite with spirv-tools linked" OFF)
include_directories("./")
IF(MNN_SEP_BUILD)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../../")
    add_library(
        MNN_Vulkan
        SHARED
        ${MNN_Vulkan_SRC}
        "${CMAKE_CURRENT_LIST_DIR}/compiler/AllShader.cpp"
        )
    target_link_libraries(MNN_Vulkan MNN)
ELSE()
    add_library(
        MNN_Vulkan
        OBJECT
        ${MNN_Vulkan_SRC}
        "${CMAKE_CURRENT_LIST_DIR}/compiler/AllShader.cpp"
        )
ENDIF()
target_include_directories(MNN_Vulkan PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include/" "${CMAKE_CURRENT_LIST_DIR}/component/" "${CMAKE_CURRENT_LIST_DIR}/runtime" "${CMAKE_CURRENT_LIST_DIR}/execution" "${CMAKE_CURRENT_LIST_DIR}/backend")
if (CMAKE_SYSTEM_NAME MATCHES "^Android")
  add_definitions(-DVK_USE_PLATFORM_ANDROID_KHR)
endif()
IF(MNN_VULKAN_REGEN)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_LIST_DIR}/compiler/AllShader.cpp"
    COMMAND ${PYTHON_EXECUTABLE}
    "${CMAKE_CURRENT_LIST_DIR}/compiler/makeshader.py"
    "${CMAKE_CURRENT_LIST_DIR}/execution/glsl/"
    "${CMAKE_SOURCE_DIR}/include/MNN/backend/vulkan/shaders/AllShader.h"
    "${CMAKE_CURRENT_LIST_DIR}/compiler/AllShader.cpp"
    COMMENT "Vulkan Code Generation"
  )
  add_custom_target (MNN_VulkanCodeGen DEPENDS "${CMAKE_CURRENT_LIST_DIR}/compiler/AllShader.cpp")
  add_dependencies(MNN_Vulkan MNN_VulkanCodeGen)
ENDIF()


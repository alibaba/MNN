FILE(GLOB_RECURSE MNN_Vulkan_SRC ${CMAKE_CURRENT_LIST_DIR}/*)
option(MNN_VULKAN_REGEN "Regenerate Vulkan Shader binaries. Requires FULL glslang suite with spirv-tools linked" OFF)
include_directories("./")
option(MNN_VULKAN_SUPPORT_COMPABILITY "Support high precision for Vulkan" OFF)
if (MNN_VULKAN_SUPPORT_COMPABILITY)
    add_definitions(-DMNN_VULKAN_SUPPORT_COMPABILITY)
endif()
if(MNN_USE_SYSTEM_LIB)
    find_package(Vulkan REQUIRED)
    set(MNN_VULKAN_LIBS ${Vulkan_LIBRARIES} PARENT_SCOPE)
else()
    add_definitions(-DMNN_USE_LIB_WRAPPER)
endif()
IF(MNN_SEP_BUILD)
    if (MNN_BUILD_FOR_ANDROID_COMMAND)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../../")
    endif()
    add_library(
        MNN_Vulkan
        SHARED
        ${MNN_Vulkan_SRC}
        "${CMAKE_CURRENT_LIST_DIR}/compiler/AllShader.cpp"
        )
    target_link_libraries(MNN_Vulkan MNN)
    if (MNN_USE_SYSTEM_LIB)
        target_link_libraries(MNN_Vulkan ${Vulkan_LIBRARY})
    else()
        target_link_libraries(MNN_Vulkan dl)
    endif()
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


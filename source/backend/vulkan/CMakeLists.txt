if(MNN_VULKAN)
  FILE(GLOB_RECURSE MNN_Vulkan_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
  option(MNN_VULKAN_REGEN "Regenerate Vulkan Shader binaries. Requires FULL glslang suite with spirv-tools linked" OFF)
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
    add_custom_target (MNNVulkanCodeGen DEPENDS "${CMAKE_CURRENT_LIST_DIR}/compiler/AllShader.cpp")
  ENDIF()
  add_library(
      MNNVulkan
      OBJECT
      ${MNN_Vulkan_SRC}
      "${CMAKE_CURRENT_LIST_DIR}/compiler/AllShader.cpp"

  )
  target_include_directories(MNNVulkan PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include/" "${CMAKE_CURRENT_LIST_DIR}/component/" "${CMAKE_CURRENT_LIST_DIR}/runtime" "${CMAKE_CURRENT_LIST_DIR}/execution" "${CMAKE_CURRENT_LIST_DIR}/backend")
  IF(MNN_VULKAN_REGEN)
    add_dependencies(MNNVulkan MNNVulkanCodeGen)
  ENDIF()
  list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNVulkan>)
  list(APPEND MNN_TARGETS MNNVulkan)
  SET(MNN_OBJECTS_TO_LINK "${MNN_OBJECTS_TO_LINK}" PARENT_SCOPE)
  SET(MNN_TARGETS "${MNN_TARGETS}" PARENT_SCOPE)
endif()

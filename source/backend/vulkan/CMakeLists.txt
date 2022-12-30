option(MNN_VULKAN_IMAGE "Use Image as basic type" ON)
if (MNN_VULKAN_IMAGE)
    FILE(GLOB_RECURSE MNN_Vulkan_SRC ${CMAKE_CURRENT_LIST_DIR}/image/* ${CMAKE_CURRENT_LIST_DIR}/component/* ${CMAKE_CURRENT_LIST_DIR}/runtime/* ${CMAKE_CURRENT_LIST_DIR}/vulkan/*)
else()
    FILE(GLOB_RECURSE MNN_Vulkan_SRC ${CMAKE_CURRENT_LIST_DIR}/buffer/* ${CMAKE_CURRENT_LIST_DIR}/component/* ${CMAKE_CURRENT_LIST_DIR}/runtime/* ${CMAKE_CURRENT_LIST_DIR}/vulkan/*)
endif()

include_directories("./")
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
        )
ENDIF()

include_directories("${CMAKE_CURRENT_LIST_DIR}/include/" "${CMAKE_CURRENT_LIST_DIR}/component/" "${CMAKE_CURRENT_LIST_DIR}/runtime")
if (MNN_VULKAN_IMAGE)
    include_directories("${CMAKE_CURRENT_LIST_DIR}/image/execution" "${CMAKE_CURRENT_LIST_DIR}/image/backend" "${CMAKE_CURRENT_LIST_DIR}/image" "${CMAKE_CURRENT_LIST_DIR}/image/shaders")
else()
    include_directories("${CMAKE_CURRENT_LIST_DIR}/buffer/execution" "${CMAKE_CURRENT_LIST_DIR}/buffer/backend" "${CMAKE_CURRENT_LIST_DIR}/buffer" "${CMAKE_CURRENT_LIST_DIR}/buffer/shaders")
endif()

if (CMAKE_SYSTEM_NAME MATCHES "^Android")
  add_definitions(-DVK_USE_PLATFORM_ANDROID_KHR)
endif()

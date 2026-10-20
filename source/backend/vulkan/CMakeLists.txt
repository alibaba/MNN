option(MNN_VULKAN_IMAGE "Use Image as basic type" ON)
option(MNN_VULKAN_ARM_NX "Experimental ARM NX MatMul (buffer backend)" OFF)
if(MNN_VULKAN_ARM_NX)
    if(MNN_VULKAN_IMAGE)
        message(FATAL_ERROR "MNN_VULKAN_ARM_NX requires MNN_VULKAN_IMAGE=OFF")
    endif()
    find_package(Vulkan REQUIRED)
    find_path(MNN_SPIRV_HEADERS spirv/unified1/TOSA.001000.1.h REQUIRED)
    file(STRINGS "${Vulkan_INCLUDE_DIRS}/vulkan/vulkan_core.h" MNN_ARM_NX_GRAPH_HEADER
         REGEX "^#define VK_ARM_data_graph ")
    if(NOT MNN_ARM_NX_GRAPH_HEADER)
        message(FATAL_ERROR "ARM NX requires recent Vulkan-Headers with VK_ARM_data_graph")
    endif()
    # Keep the bundled headers for ordinary builds; this opt-in path needs the ML extensions.
    include_directories(BEFORE ${Vulkan_INCLUDE_DIRS} ${MNN_SPIRV_HEADERS})
    add_definitions(-DMNN_VULKAN_ARM_NX)
endif()
if (MNN_VULKAN_IMAGE)
    FILE(GLOB_RECURSE MNN_Vulkan_SRC ${CMAKE_CURRENT_LIST_DIR}/image/* ${CMAKE_CURRENT_LIST_DIR}/component/* ${CMAKE_CURRENT_LIST_DIR}/runtime/* ${CMAKE_CURRENT_LIST_DIR}/vulkan/*)
else()
    FILE(GLOB_RECURSE MNN_Vulkan_SRC ${CMAKE_CURRENT_LIST_DIR}/buffer/* ${CMAKE_CURRENT_LIST_DIR}/component/* ${CMAKE_CURRENT_LIST_DIR}/runtime/* ${CMAKE_CURRENT_LIST_DIR}/vulkan/*)
endif()

if(MNN_GPU_TIME_PROFILE)
    add_definitions(-DENABLE_VULKAN_TIME_PROFILE)
    IF (MNN_GPU_PROFILE_SILENT)
        add_definitions(-DMNN_GPU_PROFILE_SILENT)
    ENDIF()
ENDIF()


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
        target_link_libraries(MNN_Vulkan ${CMAKE_DL_LIBS})
    endif()
ELSE()
    add_library(
        MNN_Vulkan
        OBJECT
        ${MNN_Vulkan_SRC}
        )
ENDIF()

include_directories("${CMAKE_CURRENT_LIST_DIR}/include/" "${CMAKE_CURRENT_LIST_DIR}/component/" "${CMAKE_CURRENT_LIST_DIR}/runtime" "${CMAKE_CURRENT_LIST_DIR}/schema/current")
if (MNN_VULKAN_IMAGE)
    include_directories("${CMAKE_CURRENT_LIST_DIR}/image/execution" "${CMAKE_CURRENT_LIST_DIR}/image/backend" "${CMAKE_CURRENT_LIST_DIR}/image" "${CMAKE_CURRENT_LIST_DIR}/image/shaders")
else()
    include_directories("${CMAKE_CURRENT_LIST_DIR}/buffer/execution" "${CMAKE_CURRENT_LIST_DIR}/buffer/backend" "${CMAKE_CURRENT_LIST_DIR}/buffer" "${CMAKE_CURRENT_LIST_DIR}/buffer/shaders")
endif()

if (CMAKE_SYSTEM_NAME MATCHES "^Android")
    add_definitions(-DVK_USE_PLATFORM_ANDROID_KHR)
endif()

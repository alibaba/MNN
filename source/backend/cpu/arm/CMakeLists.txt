IF(NOT DEFINED ARCHS)
  set(ARCHS "")
ENDIF()
FILE(GLOB MNN_AArch32_SRC ${CMAKE_CURRENT_LIST_DIR}/arm32/*.s ${CMAKE_CURRENT_LIST_DIR}/arm32/*.S)
FILE(GLOB MNN_AArch64_SRC ${CMAKE_CURRENT_LIST_DIR}/arm64/*.s ${CMAKE_CURRENT_LIST_DIR}/arm64/*.S)


# remove the armv82 extension assemblies file
if(NOT MNN_ARM82)
    list(REMOVE_ITEM MNN_AArch64_SRC ${CMAKE_CURRENT_LIST_DIR}/arm64/MNNGemmInt8AddBiasScale_ARMV82_Unit.S)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7" OR ARCHS MATCHES "^armv7(;armv7s)?")
    message(STATUS "Enabling AArch32 Assemblies")
    add_library(MNNARM32 OBJECT ${MNN_AArch32_SRC})
    target_include_directories(MNNARM32 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNARM32>)
    list(APPEND MNN_TARGETS MNNARM32)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR ARCHS STREQUAL "arm64")
    message(STATUS "Enabling AArch64 Assemblies")
    add_library(MNNARM64 OBJECT ${MNN_AArch64_SRC})
    target_include_directories(MNNARM64 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNARM64>)
    list(APPEND MNN_TARGETS MNNARM64)

    if(MNN_ARM82)
        message(STATUS "Enable INT8 SDOT")
        # add_definitions(-DENABLE_ARMV82)
        target_compile_options(MNNARM64 PRIVATE -march=armv8.2-a+dotprod)
    endif()

else()
# Building fat binary requires multiple seperate builds and lipo-by-hand under CMake's design
endif()

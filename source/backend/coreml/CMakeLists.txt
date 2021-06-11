# CustomLayer Metal Impl
file(GLOB MNN_COREML_METAL_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/backend/*.metal
)

IF(DEFINED SDK_VERSION)
    #Defined by iOS toolchain
    SET(METAL_SDK_PLAT "iphoneos")
ELSE()
    SET(METAL_SDK_PLAT "macosx")
ENDIF()

message(STATUS "Compiling CoreML Metal Kernels with ${METAL_SDK_PLAT} SDK")

message(STATUS "Generating coreml.metallib at ${PROJECT_BINARY_DIR}/coreml.metallib")

add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/coreml.metallib
                  COMMAND xcrun -sdk ${METAL_SDK_PLAT}
                  metal "${MNN_COREML_METAL_SRCS}"
                  -o ${PROJECT_BINARY_DIR}/coreml.metallib
                  COMMAND_EXPAND_LISTS)

add_custom_target(MNNCoreMLMetalLib DEPENDS
                 ${PROJECT_BINARY_DIR}/coreml.metallib
                 COMMENT "Generating coreml.metallib")

# CoreML
file(GLOB MNN_COREML_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/backend/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/*.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/*.mm
    ${CMAKE_CURRENT_LIST_DIR}/execution/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/execution/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/mlmodel/src/*.c
)

add_library(
    MNNCoreML 
    STATIC
    ${MNN_COREML_SRCS}
    ${MNNCoreMLMetalLib}
)

target_include_directories(MNNCoreML PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/mlmodel/include
    ${CMAKE_CURRENT_LIST_DIR}/backend
    ${CMAKE_CURRENT_LIST_DIR}/execution
)

add_dependencies(MNNCoreML MNNCoreMLMetalLib)

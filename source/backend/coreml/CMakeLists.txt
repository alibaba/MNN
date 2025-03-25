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
    OBJECT
    ${MNN_COREML_SRCS}
)
set_property(TARGET MNNCoreML APPEND_STRING PROPERTY COMPILE_FLAGS "-fobjc-arc")

target_include_directories(MNNCoreML PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/mlmodel/include
    ${CMAKE_CURRENT_LIST_DIR}/backend
    ${CMAKE_CURRENT_LIST_DIR}/execution
)


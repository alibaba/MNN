# Android NN API
file(GLOB MNN_NNAPI_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/backend/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/execution/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/execution/*.hpp
)

add_library(
    MNN_NNAPI 
    OBJECT
    ${MNN_NNAPI_SRCS}
)

IF (MNN_ARM82)
    target_compile_options(MNN_NNAPI PRIVATE -DMNN_NNAPI_ENABLED=1 -DENABLE_ARMV82)
ELSE()
    target_compile_options(MNN_NNAPI PRIVATE -DMNN_NNAPI_ENABLED=1)
ENDIF()

target_include_directories(MNN_NNAPI PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/backend
    ${CMAKE_CURRENT_LIST_DIR}/execution
)

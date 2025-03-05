# Android NN API
file(GLOB MNN_NeuronAdapter_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/backend/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/execution/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/execution/*.hpp
)

add_library(
    MNN_NeuronAdapter
    OBJECT
    ${MNN_NeuronAdapter_SRCS}
)

IF (MNN_ARM82)
    target_compile_options(MNN_NeuronAdapter PRIVATE -DMNN_NeuronAdapter_ENABLED=1 -DENABLE_ARMV82)
ELSE()
    target_compile_options(MNN_NeuronAdapter PRIVATE -DMNN_NeuronAdapter_ENABLED=1)
ENDIF()

target_include_directories(MNN_NeuronAdapter PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/backend
    ${CMAKE_CURRENT_LIST_DIR}/execution
)

# Android NN API
file(GLOB MNN_NNAPI_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/backend/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/execution/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/execution/*.hpp
)

add_library(
    MNN_NNAPI 
    STATIC
    ${MNN_NNAPI_SRCS}
)

target_compile_options(MNN_NNAPI PRIVATE -DMNN_NNAPI_ENABLED=1)

target_include_directories(MNN_NNAPI PRIVATE 
    ${CMAKE_CURRENT_LIST_DIR}/backend
    ${CMAKE_CURRENT_LIST_DIR}/execution
)
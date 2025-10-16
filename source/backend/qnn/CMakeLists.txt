option(MNN_QNN_CONVERT_MODE "Enable the Convert mode of the QNN backend." OFF)

file(GLOB BACKEND_SRCS ${CMAKE_CURRENT_LIST_DIR}/backend/*.cpp)
file(GLOB EXECUTION_SRCS ${CMAKE_CURRENT_LIST_DIR}/execution/*.cpp)
set(MNN_QNN_SRCS ${BACKEND_SRCS} ${EXECUTION_SRCS})

if (MNN_QNN_CONVERT_MODE)
    file(GLOB CONVERTOR_SRCS ${CMAKE_CURRENT_LIST_DIR}/convertor/*.cpp)
    list(APPEND MNN_QNN_SRCS ${CONVERTOR_SRCS})
endif()

add_library(MNN_QNN OBJECT ${MNN_QNN_SRCS})

if (MNN_ARM82)
    target_compile_options(MNN_QNN PRIVATE -DENABLE_ARMV82)
endif()

if (MNN_QNN_CONVERT_MODE)
    target_compile_options(MNN_QNN PRIVATE -DENABLE_QNN_CONVERT_MODE)
endif()

if (MNN_QNN_ONLINE_FINALIZE)
    target_compile_options(MNN_QNN PRIVATE -DENABLE_QNN_ONLINE_FINALIZE)
endif()

target_include_directories(MNN_QNN PRIVATE ${CMAKE_CURRENT_LIST_DIR}/backend/)
target_include_directories(MNN_QNN PRIVATE ${CMAKE_CURRENT_LIST_DIR}/convertor/)
target_include_directories(MNN_QNN PRIVATE ${CMAKE_CURRENT_LIST_DIR}/3rdParty/include/)

file(GLOB_RECURSE MNN_QNN_SRCS ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

add_library(MNN_QNN OBJECT ${MNN_QNN_SRCS})

if (MNN_ARM82)
    target_compile_options(MNN_QNN PRIVATE -DENABLE_ARMV82)
endif()

target_include_directories(MNN_QNN PRIVATE ${CMAKE_CURRENT_LIST_DIR}/backend/)
target_include_directories(MNN_QNN PRIVATE ${CMAKE_CURRENT_LIST_DIR}/3rdParty/include/)

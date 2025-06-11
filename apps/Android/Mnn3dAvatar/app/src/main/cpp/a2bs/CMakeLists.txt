cmake_minimum_required(VERSION 3.4.1)

project(mnn_a2bs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#include_directories(
#        ${CMAKE_CURRENT_LIST_DIR}/include
#        ${CMAKE_CURRENT_LIST_DIR}/include/a2bs
#        ${THIRD_PARTY_INCLUDES_DIR}
#        ${PROJECT_INCLUDE_DIR}
#        ${COMMON_DIR}/include
#)

#MESSAGE(FATAL_ERROR "PROJECT_NAME: ${PROJECT_NAME}/include")


FILE(GLOB SOURCE_FILES
        ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

#set(SOURCE_FILES
#        src/audio_to_flame_blend_shape.cpp
#        src/audio_to_3dgs_blend_shape.cpp
#        src/gs_body_converter.cpp
#        src/utils.cpp
#        src/pixelai_audio2bs_logger.cpp
#)


add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${THIRD_PARTY_INCLUDES_DIR}
        ${PROJECT_INCLUDE_DIR}
        ${COMMON_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/include/a2bs
        ${CMAKE_CURRENT_LIST_DIR}/include
        PRIVATE
)
target_link_libraries(${PROJECT_NAME}
        log
        android
        ${MNN_LIBRARIES}
)
get_target_property(INCLUDE_DIRS ${PROJECT_NAME} INCLUDE_DIRECTORIES)
#MESSAGE(FATAL_ERROR "Target_include directories: ${INCLUDE_DIRS}")

set_target_properties(${PROJECT_NAME} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include"
)
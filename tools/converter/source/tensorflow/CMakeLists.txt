IF(MNN_BUILD_PROTOBUFFER)
    file(GLOB TENSORFLOW_PROTO_SRCS ${CMAKE_CURRENT_LIST_DIR}/generated/*)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/generated/)
ELSE()
    protobuf_generate_cpp(TENSORFLOW_PROTO_SRCS TENSORFLOW_PROTO_HDRS
        ${CMAKE_CURRENT_LIST_DIR}/attr_value.proto
        ${CMAKE_CURRENT_LIST_DIR}/function.proto
        ${CMAKE_CURRENT_LIST_DIR}/graph.proto
        ${CMAKE_CURRENT_LIST_DIR}/node_def.proto
        ${CMAKE_CURRENT_LIST_DIR}/op_def.proto
        ${CMAKE_CURRENT_LIST_DIR}/resource_handle.proto
        ${CMAKE_CURRENT_LIST_DIR}/tensor.proto
        ${CMAKE_CURRENT_LIST_DIR}/tensor_shape.proto
        ${CMAKE_CURRENT_LIST_DIR}/types.proto
        ${CMAKE_CURRENT_LIST_DIR}/versions.proto
    )
ENDIF()

file(GLOB TENSORFLOW_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
add_library(MNNConverterTF OBJECT ${TENSORFLOW_SRC} ${TENSORFLOW_PROTO_SRCS})
target_include_directories(MNNConverterTF PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/)
add_dependencies(MNNConverterTF MNNCompress)
list(APPEND MNN_CONVERTER_BACKENDS_OBJECTS $<TARGET_OBJECTS:MNNConverterTF>)
list(APPEND MNN_CONVERTER_BACKENDS_TARGETS MNNConverterTF)

protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS
    ${CMAKE_CURRENT_LIST_DIR}/onnx.proto
    ${CMAKE_CURRENT_LIST_DIR}/onnx-operators.proto
    # onnx-ml.proto
    # onnx-operators-ml.proto
)

file(GLOB ONNX_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_library(MNNConverterONNX OBJECT ${ONNX_SRC} ${ONNX_PROTO_SRCS})
target_include_directories(MNNConverterONNX PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)
list(APPEND MNN_CONVERTER_BACKENDS_OBJECTS $<TARGET_OBJECTS:MNNConverterONNX>)
list(APPEND MNN_CONVERTER_BACKENDS_TARGETS MNNConverterONNX)

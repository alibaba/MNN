protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS
    ${CMAKE_CURRENT_LIST_DIR}/onnx.proto
    ${CMAKE_CURRENT_LIST_DIR}/onnx-operators.proto
    # onnx-ml.proto
    # onnx-operators-ml.proto
)

file(GLOB ONNX_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_library(MNNConverterONNX OBJECT ${ONNX_SRC} ${ONNX_PROTO_SRCS})
message(STATUS "Onnx: ${ONNX_PROTO_HDRS}")
add_executable(OnnxClip ${CMAKE_CURRENT_LIST_DIR}/exec/OnnxClip.cpp ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS} ${CMAKE_CURRENT_LIST_DIR}/OnnxUtils.cpp)
target_link_libraries(OnnxClip ${Protobuf_LIBRARIES})
list(APPEND MNN_CONVERTER_BACKENDS_OBJECTS $<TARGET_OBJECTS:MNNConverterONNX>)
list(APPEND MNN_CONVERTER_BACKENDS_TARGETS MNNConverterONNX)

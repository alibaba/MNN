IF(MNN_BUILD_PROTOBUFFER)
    file(GLOB ONNX_PROTO_SRCS ${CMAKE_CURRENT_LIST_DIR}/generated/*)
    include_directories(${CMAKE_CURRENT_LIST_DIR}/generated/)
ELSE()
protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS
    ${CMAKE_CURRENT_LIST_DIR}/onnx.proto
    # onnx-operators.proto
    # onnx-ml.proto
    # onnx-operators-ml.proto
)
ENDIF()
file(GLOB ONNX_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_library(MNNConverterONNX OBJECT ${ONNX_SRC} ${ONNX_PROTO_SRCS})
add_dependencies(MNNConverterONNX MNNCompress)
message(STATUS "Onnx: ${ONNX_PROTO_HDRS}")

add_executable(OnnxClip ${CMAKE_CURRENT_LIST_DIR}/exec/OnnxClip.cpp ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS} ${CMAKE_CURRENT_LIST_DIR}/OnnxUtils.cpp)
target_link_libraries(OnnxClip ${Protobuf_LIBRARIES})

list(APPEND MNN_CONVERTER_BACKENDS_OBJECTS $<TARGET_OBJECTS:MNNConverterONNX>)
list(APPEND MNN_CONVERTER_BACKENDS_TARGETS MNNConverterONNX)

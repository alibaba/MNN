file(GLOB_RECURSE OPTIMIZER_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_library(MNNConverterOpt OBJECT ${OPTIMIZER_SRC})
# MNN_compression.ph.h
add_dependencies(MNNConverterOpt MNNCompress)
target_include_directories(MNNConverterOpt PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/)
list(APPEND MNN_CONVERTER_BACKENDS_OBJECTS $<TARGET_OBJECTS:MNNConverterOpt>)
list(APPEND MNN_CONVERTER_BACKENDS_TARGETS MNNConverterOpt)

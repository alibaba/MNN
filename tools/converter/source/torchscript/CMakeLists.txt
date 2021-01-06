file(GLOB TS_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_library(MNNConverterTS OBJECT ${TS_SRC})

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
find_package(Torch REQUIRED)

include(${LLVM_DIR}/TableGen.cmake)
include(${MLIR_DIR}/AddMLIR.cmake)

# add include dir
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_LIST_DIR}/MNNOps.td)
mlir_tablegen(MNNOps.h.inc -gen-op-decls)
mlir_tablegen(MNNOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MNNOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_LIST_DIR}/TorchScriptOps.td)
mlir_tablegen(TorchScriptOps.h.inc -gen-op-decls)
mlir_tablegen(TorchScriptOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TorchScriptOpsIncGen)

add_dependencies(MNNConverterTS
  TorchScriptOpsIncGen
  MNNOpsIncGen
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

list(APPEND MNN_CONVERTER_TS_LIBS MLIRAnalysis)
list(APPEND MNN_CONVERTER_TS_LIBS MLIRIR)
list(APPEND MNN_CONVERTER_TS_LIBS MLIRParser)
list(APPEND MNN_CONVERTER_TS_LIBS MLIRPass)
list(APPEND MNN_CONVERTER_TS_LIBS MLIRSideEffectInterfaces)
list(APPEND MNN_CONVERTER_TS_LIBS MLIRTransforms)
list(APPEND MNN_CONVERTER_TS_LIBS ${TORCH_LIBRARIES})
# target_link_libraries(MNNConverterTS
#   PUBLIC
#     MLIRAnalysis
#     MLIRIR
#     MLIRParser
#     MLIRPass
#     MLIRSideEffectInterfaces
#     MLIRTransforms
#     ${TORCH_LIBRARIES})
# set_target_properties(MNNConverterTS PROPERTIES CXX_STANDARD 17 CXX_VISIBILITY_PRESET hidden)
set_target_properties(MNNConverterTS PROPERTIES CXX_STANDARD 17)

list(APPEND MNN_CONVERTER_BACKENDS_OBJECTS $<TARGET_OBJECTS:MNNConverterTS>)
list(APPEND MNN_CONVERTER_BACKENDS_TARGETS MNNConverterTS)

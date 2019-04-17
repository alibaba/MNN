# put output to build dir
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../)
message(STATUS ${CMAKE_CURRENT_BINARY_DIR})
SET(REVERT_PATH ../tools/cpp)
include_directories(${REVERT_PATH})
include_directories(../tools)

add_executable(benchmark.out benchmark.cpp ${REVERT_PATH}/revertMNNModel.cpp)
target_link_libraries(benchmark.out ${MNN_DEPEND})

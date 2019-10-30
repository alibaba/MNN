# put output to build dir
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../)
message(STATUS ${CMAKE_CURRENT_BINARY_DIR})

add_executable(benchmark.out benchmark.cpp ../tools/cpp/revertMNNModel.cpp)
target_include_directories(benchmark.out PRIVATE "../tools/cpp" "../tools")
target_link_libraries(benchmark.out ${MNN_DEPEND})

file(GLOB_RECURSE SRC_FILES ./exprModels/*.cpp)
add_executable(benchmarkExprModels.out benchmarkExprModels.cpp ${SRC_FILES})
target_include_directories(benchmarkExprModels.out PRIVATE "./exprModels" "../" "../include")
target_link_libraries(benchmarkExprModels.out ${MNN_DEPEND} MNN_Express)

# put output to build dir
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../)

file(GLOB_RECURSE Files "*.cpp")
include_directories(".")
include_directories("express/include")
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    file(GLOB_RECURSE MMFiles "*.mm")
    add_executable(run_test.out ${Files} ${MMFiles})
    target_link_libraries(run_test.out "-framework Foundation")
else()
    add_executable(run_test.out ${Files})
endif()
target_link_libraries(run_test.out ${MNN_DEPEND} MNN_Express)

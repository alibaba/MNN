file(GLOB QUANFILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
add_executable(quantized.out ${QUANFILES})
target_link_libraries(quantized.out ${MNN_DEPS})

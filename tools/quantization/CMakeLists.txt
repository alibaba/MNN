file(GLOB QUANFILES ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.hpp)
add_executable(quantized.out ${QUANFILES})
target_link_libraries(quantized.out ${MNN_DEPS})

file(GLOB QUANOPTFILES ${CMAKE_CURRENT_LIST_DIR}/opttools/*)
add_executable(quantized_model_optimize.out ${QUANOPTFILES})
target_link_libraries(quantized_model_optimize.out ${MNN_DEPS})

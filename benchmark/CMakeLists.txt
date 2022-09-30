

set (EXTRA_INCLUDE "")
set (EXTRA_LIBS "")
set (EXTRA_OPTIONS "")

if(MNN_CUDA_PROFILE)
    set(CUDA_MIN_VERSION "9.0")
    find_package(CUDA ${CUDA_MIN_VERSION})
    MESSAGE("benchmark message:CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
    link_directories(${CUDA_INCLUDE_DIRS}/../lib/
        ${CUDA_INCLUDE_DIRS}/../lib64/)
    set(EXTRA_INCLUDE ${CUDA_INCLUDE_DIRS})
    set(EXTRA_LIBS  -lnvToolsExt)
    set(EXTRA_OPTIONS -DMNN_CUDA_PROFILE)
endif()


add_executable(benchmark.out ${CMAKE_CURRENT_LIST_DIR}/benchmark.cpp ${CMAKE_CURRENT_SOURCE_DIR}/tools/cpp/revertMNNModel.cpp)
target_include_directories(benchmark.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools/cpp/ ${CMAKE_CURRENT_SOURCE_DIR}/tools/ ${EXTRA_INCLUDE})
target_compile_options(benchmark.out PRIVATE ${EXTRA_OPTIONS})
target_link_libraries(benchmark.out ${MNN_DEPS}  ${EXTRA_LIBS})

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/exprModels/*.cpp)
add_executable(benchmarkExprModels.out ${CMAKE_CURRENT_LIST_DIR}/benchmarkExprModels.cpp ${SRC_FILES})
target_include_directories(benchmarkExprModels.out PRIVATE "${CMAKE_CURRENT_LIST_DIR}/exprModels" ${CMAKE_CURRENT_SOURCE_DIR}/)
target_link_libraries(benchmarkExprModels.out ${MNN_DEPS})

if (MSVC AND NOT MNN_BUILD_SHARED_LIBS)
  foreach (DEPEND ${MNN_DEPS})
    target_link_options(benchmark.out PRIVATE /WHOLEARCHIVE:$<TARGET_FILE:${DEPEND}>)
    target_link_options(benchmarkExprModels.out PRIVATE /WHOLEARCHIVE:$<TARGET_FILE:${DEPEND}>)
  endforeach ()
endif()
# Keep TORQ-Tile's C++17 and RVV requirements private to its objects.
set(MNN_TORQ_TILE_ROOT ${PROJECT_SOURCE_DIR}/3rd_party/torq-tile)
add_library(MNNTorqTileKernels OBJECT
    ${MNN_TORQ_TILE_ROOT}/src/gemm/gemm_f32_f32/sgemm_f32_f32t_f32/tqt_sgemm_f32_f32t_f32_8x3vl_rvv.cpp
)
set_target_properties(MNNTorqTileKernels PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_include_directories(MNNTorqTileKernels PRIVATE ${MNN_TORQ_TILE_ROOT})

add_library(MNNTorqTile OBJECT
    ${CMAKE_CURRENT_LIST_DIR}/MNNTorqTileMatMul.cpp
    ${CMAKE_CURRENT_LIST_DIR}/MNNTorqTileFunctions.cpp
)
target_include_directories(MNNTorqTile PRIVATE ${MNN_TORQ_TILE_ROOT})
target_compile_definitions(MNNCPU PRIVATE MNN_USE_TORQ_TILE)
foreach(TORQ_TARGET MNNTorqTile MNNTorqTileKernels)
    target_compile_options(${TORQ_TARGET} PRIVATE -march=${MNN_RVV_BASE_MARCH} -mabi=lp64d)
    if (MNN_RVV_MCPU)
        target_compile_options(${TORQ_TARGET} PRIVATE -mcpu=${MNN_RVV_MCPU})
    endif()
    if (MNN_RVV_MTUNE)
        target_compile_options(${TORQ_TARGET} PRIVATE -mtune=${MNN_RVV_MTUNE})
    endif()
    if (MNN_RVV_FAST_MATH)
        target_compile_options(${TORQ_TARGET} PRIVATE -ffast-math)
    endif()
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:${TORQ_TARGET}>)
    list(APPEND MNN_TARGETS ${TORQ_TARGET})
endforeach()

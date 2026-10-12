# A single FP32 8x4 kernel and its MNN adapter, built as standard RVV C++11.
add_library(MNNTorqTile OBJECT
    ${CMAKE_CURRENT_LIST_DIR}/MNNTorqTileKernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/MNNTorqTileMatMul.cpp
    ${CMAKE_CURRENT_LIST_DIR}/MNNTorqTileFunctions.cpp
)
target_compile_definitions(MNNCPU PRIVATE MNN_USE_TORQ_TILE)
target_compile_options(MNNTorqTile PRIVATE -march=${MNN_RVV_BASE_MARCH} -mabi=lp64d)
if (MNN_RVV_MCPU)
    target_compile_options(MNNTorqTile PRIVATE -mcpu=${MNN_RVV_MCPU})
endif()
if (MNN_RVV_MTUNE)
    target_compile_options(MNNTorqTile PRIVATE -mtune=${MNN_RVV_MTUNE})
endif()
if (MNN_RVV_FAST_MATH)
    target_compile_options(MNNTorqTile PRIVATE -ffast-math)
endif()
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNTorqTile>)
list(APPEND MNN_TARGETS MNNTorqTile)

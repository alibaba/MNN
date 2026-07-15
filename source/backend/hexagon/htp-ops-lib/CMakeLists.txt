cmake_minimum_required(VERSION 3.14.3)
project(htp_ops C CXX ASM)
# cmake_policy(SET CMP0115 OLD)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(MNN_HTP_OPS_STUB_NAME MNN_htpops)
set(MNN_HTP_OPS_SKEL_NAME MNN_htpops_skel)

if(HEXAGON_SDK_ROOT)
    include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
else()
    include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)
endif()
set(ignoreVar ${DEPENDENCIES}${V})

# NOTE(hzx): QAIC generated code are located in ${CMAKE_CURRENT_BINARY_DIR}
set(common_incs
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../3rd_party/flatbuffers/include/
    ${CMAKE_CURRENT_BINARY_DIR}
    ${HEXAGON_SDK_ROOT}/incs/
    ${HEXAGON_SDK_ROOT}/incs/stddef/
    ${HEXAGON_SDK_ROOT}/rtos/qurt/
    ${HEXAGON_SDK_ROOT}/utils/examples/
)
include_directories(${common_incs})

if(${OS_TYPE} MATCHES "HLOS")
    # HLOS dynamic binary
    add_library(htp_ops SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/htp_ops_stub.c
        ${HEXAGON_SDK_ROOT}/utils/examples/dsp_capabilities_utils.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/op_export.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/session.c
    )
    set_target_properties(htp_ops PROPERTIES OUTPUT_NAME ${MNN_HTP_OPS_STUB_NAME})

    # Build interface files
    build_idl(include/htp_ops.idl htp_ops)

    # Find the DSP from provided domain value, default is CDSP
    choose_dsprpc(${DSP_TYPE} dsprpc)
    link_custom_library(htp_ops ${dsprpc})

    # Copy binaries from ship to CMAKE_CURRENT_BINARY_DIR folder
    copy_binaries(htp_ops)

else()

    # Shared skel library
    add_library(htp_ops_skel SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/htp_ops_skel.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/commu.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/execute_command.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/blit_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/tensor_convert.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/layer_norm_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/matmul_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/matmul_q4block_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/tmac_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/im2col_convolution_fp16.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/pool_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/conv_depthwise_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/attention_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/attention_hmx.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/attention_sync_setup.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/attention_push_kv.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/attention_sync_process.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/attention_entry.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/flash_attention_block_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/unary_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/cast_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/eltwise_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/lstm_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/topk_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/softmax_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/loop_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/rope_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/scale_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/shared_gather_ops.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/power.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/hmx_mgr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/worker_pool.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/mmap_mgr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/vtcm_mgr.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/matmul_q4fp16.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/matmul_q4fp16_mle32.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/matmul_q4block_fp16_mle32.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/tmac_a16w1_fp16.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/pool_fp16.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dsp/ops/depthwise_conv_fp16.c
    )
    set_target_properties(htp_ops_skel PROPERTIES OUTPUT_NAME ${MNN_HTP_OPS_SKEL_NAME})
    build_idl(include/htp_ops.idl htp_ops_skel)

    # Enable HMX inline assembly
    set(HTP_OPS_DSP_FLAGS "-mhmx")
    string(REGEX REPLACE "^v" "" HTP_OPS_DSP_ARCH_NUMBER "${DSP_ARCH}")
    if(HTP_OPS_DSP_ARCH_NUMBER LESS 79)
        set(HTP_OPS_DSP_FLAGS "${HTP_OPS_DSP_FLAGS} -mhvx-ieee-fp")
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${HTP_OPS_DSP_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HTP_OPS_DSP_FLAGS}")

    # Link C++ libraries
    target_link_libraries(htp_ops_skel ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++abi.so.1)
    target_link_libraries(htp_ops_skel ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++.so.1)

    # Copy binaries from ship to CMAKE_CURRENT_BINARY_DIR folder
    copy_binaries(htp_ops_skel)

endif()

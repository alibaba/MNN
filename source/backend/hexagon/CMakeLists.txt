# source/backend/hexagon/CMakeLists.txt

if(NOT MNN_HEXAGON)
    return()
endif()

file(GLOB HEXAGON_BACKEND_SRCS ${CMAKE_CURRENT_LIST_DIR}/backend/*.cpp ${CMAKE_CURRENT_LIST_DIR}/backend/*.cc ${CMAKE_CURRENT_LIST_DIR}/backend/HexagonCommand.cpp)
file(GLOB HEXAGON_EXECUTION_SRCS ${CMAKE_CURRENT_LIST_DIR}/execution/*.cpp)

set(MNN_HEXAGON_SRCS
    ${HEXAGON_BACKEND_SRCS}
    ${HEXAGON_EXECUTION_SRCS}
)

add_library(MNN_Hexagon OBJECT ${MNN_HEXAGON_SRCS})

# MNN_HEXAGON_ENABLED is a global add_definitions in the root CMakeLists (the geometry needs it too).
if (MNN_HEXAGON_ASAN)
    target_compile_definitions(MNN_Hexagon PRIVATE -DMNN_HEXAGON_ASAN=1)
endif()
# App-only: holding the DSP power vote between command flushes matters for untrusted app processes;
# for command line tools the original per-flush acquire/release is kept. Off by default. Only the
# .cpp behavior is guarded, so the class layout (and thus the header) is identical either way.
if (MNN_HEXAGON_APP_POWER_HOLD)
    target_compile_definitions(MNN_Hexagon PRIVATE -DMNN_HEXAGON_APP_POWER_HOLD=1)
endif()
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" OR ANDROID_ABI STREQUAL "arm64-v8a")
    set_source_files_properties(
        ${CMAKE_CURRENT_LIST_DIR}/backend/HexagonBackend.cpp
        PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16"
    )
endif()
if (MNN_GPU_TIME_PROFILE)
target_compile_definitions(MNN_Hexagon PRIVATE -DMNN_GPU_TIME_PROFILE)
endif()

target_include_directories(MNN_Hexagon
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/backend
        ${CMAKE_CURRENT_LIST_DIR}/execution
        ${CMAKE_CURRENT_LIST_DIR}/htp-ops-lib/include
        ${CMAKE_CURRENT_LIST_DIR}
        ${_HEXAGON_ROOT}/includes
)

if(MNN_BUILD_TEST)
    add_executable(test_hexagon_quant_contract
        ${CMAKE_CURRENT_LIST_DIR}/tests/test_hexagon_quant_contract.cpp
        ${CMAKE_CURRENT_LIST_DIR}/backend/HexagonQuantization.cpp
    )
    target_include_directories(test_hexagon_quant_contract
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/backend
            ${CMAKE_CURRENT_LIST_DIR}/htp-ops-lib/include
    )

    add_executable(test_hexagon_quant_tensor_contract
        ${CMAKE_CURRENT_LIST_DIR}/tests/test_hexagon_quant_tensor_contract.cpp
    )
    target_include_directories(test_hexagon_quant_tensor_contract
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/backend
            ${CMAKE_CURRENT_LIST_DIR}/execution
            ${CMAKE_CURRENT_LIST_DIR}/htp-ops-lib/include
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/source
            ${CMAKE_SOURCE_DIR}/schema/current
            ${CMAKE_SOURCE_DIR}/3rd_party
            ${CMAKE_SOURCE_DIR}/3rd_party/flatbuffers/include
    )
    target_link_libraries(test_hexagon_quant_tensor_contract PRIVATE MNN)

    add_executable(test_hexagon_int8_abi_contract
        ${CMAKE_CURRENT_LIST_DIR}/tests/test_hexagon_int8_abi_contract.cpp
    )
    target_include_directories(test_hexagon_int8_abi_contract
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/backend
            ${CMAKE_CURRENT_LIST_DIR}/htp-ops-lib/include
    )

    add_executable(test_hexagon_command_status
        ${CMAKE_CURRENT_LIST_DIR}/tests/test_hexagon_command_status.cpp
    )
    target_include_directories(test_hexagon_command_status
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/backend
    )

    add_executable(test_hexagon_w8a8_device_contract
        ${CMAKE_CURRENT_LIST_DIR}/tests/test_hexagon_w8a8_device_contract.cpp
    )
    target_include_directories(test_hexagon_w8a8_device_contract
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/backend
            ${CMAKE_CURRENT_LIST_DIR}/execution
    )
    target_link_libraries(test_hexagon_w8a8_device_contract PRIVATE MNN)

    add_executable(test_hexagon_int8_requant_contract
        ${CMAKE_CURRENT_LIST_DIR}/htp-ops-lib/tests/int8_requant_contract_test.cpp
        ${CMAKE_CURRENT_LIST_DIR}/backend/HexagonQuantization.cpp
        ${CMAKE_CURRENT_LIST_DIR}/htp-ops-lib/src/dsp/int8_quant_ops.cc
        ${CMAKE_CURRENT_LIST_DIR}/htp-ops-lib/src/dsp/ops/int8_requant_hvx.c
    )
    target_include_directories(test_hexagon_int8_requant_contract
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/backend
            ${CMAKE_CURRENT_LIST_DIR}/htp-ops-lib/include
    )

    add_executable(test_hexagon_int8_conv_contract
        ${CMAKE_CURRENT_LIST_DIR}/htp-ops-lib/tests/int8_conv_contract_test.cpp
        ${CMAKE_CURRENT_LIST_DIR}/backend/HexagonQuantization.cpp
        ${CMAKE_CURRENT_LIST_DIR}/htp-ops-lib/src/dsp/int8_quant_ops.cc
        ${CMAKE_CURRENT_LIST_DIR}/htp-ops-lib/src/dsp/int8_conv_ops.cc
        ${CMAKE_CURRENT_LIST_DIR}/htp-ops-lib/src/dsp/ops/int8_requant_hvx.c
        ${CMAKE_CURRENT_LIST_DIR}/htp-ops-lib/src/dsp/ops/int8_conv_hvx.c
    )
    target_include_directories(test_hexagon_int8_conv_contract
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/backend
            ${CMAKE_CURRENT_LIST_DIR}/htp-ops-lib/include
    )
endif()

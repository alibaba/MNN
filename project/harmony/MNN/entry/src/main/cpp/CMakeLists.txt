cmake_minimum_required(VERSION 3.4.1)
project(Demo)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(MNN_ROOT ${NATIVERENDER_ROOT_PATH}/../../../../../../..)

set(MNN_BUILD_LLM ON CACHE BOOL "" FORCE)
set(MNN_LLM_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_LLM_OMNI ON CACHE BOOL "" FORCE)
set(LLM_SUPPORT_HTTP_RESOURCE OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(MNN_SEP_BUILD OFF CACHE BOOL "" FORCE)
set(MNN_OPENCL ON CACHE BOOL "" FORCE)
# Compiles MNN CV and image codecs into libMNN and defines LLM_SUPPORT_VISION, so Omni-class
# models can load images referenced by <img>path</img> tags in the prompt.
set(MNN_BUILD_OPENCV ON CACHE BOOL "" FORCE)
# Same codegen assumptions as the Android MnnLlmChat build: low-memory weight layout, dequant+GEMM
# fusion, fp16 (arm82) kernels and the transformer fuse path. Without these the CPU path falls back
# to the generic fp32 kernels and is several times slower.
set(MNN_ARM82 ON CACHE BOOL "" FORCE)
set(MNN_SME2 OFF CACHE BOOL "" FORCE)
set(MNN_SUPPORT_BF16 OFF CACHE BOOL "" FORCE)
set(MNN_LOW_MEMORY ON CACHE BOOL "" FORCE)
set(MNN_CPU_WEIGHT_DEQUANT_GEMM ON CACHE BOOL "" FORCE)
set(MNN_SUPPORT_TRANSFORMER_FUSE ON CACHE BOOL "" FORCE)
set(MNN_INTERNAL OFF CACHE BOOL "" FORCE)
set(MNN_WITH_PLUGIN OFF CACHE BOOL "" FORCE)
set(MNN_USE_LOGCAT ON CACHE BOOL "" FORCE)

add_subdirectory(${MNN_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/mnn)

include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include
                    ${MNN_ROOT}/include
                    ${MNN_ROOT}/transformers/llm/engine/include)

add_definitions(-Wno-format-security)
add_library(entry SHARED test.cpp llm_session.cpp llm_napi.cpp)
target_link_libraries(entry PUBLIC MNN libace_napi.z.so libhilog_ndk.z.so)

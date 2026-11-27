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
set(MNN_VULKAN ON CACHE BOOL "" FORCE)
# Compiles MNN CV and image codecs into libMNN and defines LLM_SUPPORT_VISION, so Omni-class
# models can load images referenced by <img>path</img> tags in the prompt.
set(MNN_BUILD_OPENCV ON CACHE BOOL "" FORCE)
# Builds the diffusion engine (SD/Sana/Wan) into libMNN so image generation models can run.
set(MNN_BUILD_DIFFUSION ON CACHE BOOL "" FORCE)
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
# LLM inference on HarmonyOS uses the Vulkan buffer mode only (image mode
# HANGS on -n 32 / pg512 runs on Maleoon 935 and its pp64 is 3x slower;
# buffer is the go-forward path).
set(MNN_VULKAN_IMAGE OFF CACHE BOOL "" FORCE)
# EXPERIMENT (done 2026-11-25: per-op GPU-time vs wall audit -> GPU busy 88%,
# bottleneck is kernel execution, not scheduling; observer effect slows decode):
# per-kernel timestamp timing for the Vulkan backend, dumps per-op GPU totals.
# set(MNN_GPU_TIME_PROFILE ON CACHE BOOL "" FORCE)
# EXPERIMENT (2026-11-25, done): forced OpenCL int4 weights to the buffer/SSBO
# path. Under per-kernel event profiling buffer+ALLOC_HOST_PTR kernels measure
# faster (main graph 27.6 vs 42.8ms), but end-to-end record-batch tg is 18.5
# vs image-path 20.3 — decode wall is launch-bound, kernel savings are hidden
# by the async pipeline. Default (image weights) stays optimal on Maleoon.
# GOTCHA: commenting out a `set(... CACHE FORCE)` line does NOT clear the
# cached value; rm -rf entry/.cxx/default/default/release before trusting
# any macro-toggle A/B (a stale MNN_GPU_TIME_PROFILE cache once faked a
# "buffer path collapses to tg 5.7" result).
# add_definitions(-DMNN_OPENCL_FORCE_BUFFER_WEIGHTS)
# add_definitions(-DMNN_OPENCL_NO_HOST_PTR)
# add_definitions(-DMNN_VULKAN_INT4_ILP)  # tested: tg 10.43 vs baseline 10.25, no gain (memory-side wall, not ILP)
# add_definitions(-DMNN_VULKAN_OC8_LMEM)  # WGS128 tg7.57 / WGS256 tg5.95, both < subgroup baseline 10.25
# add_definitions(-DMNN_VULKAN_DECODE_USE_GEMM)  # tested: decode via prefill GEMM(padM64) tg2.56, 4x SLOWER than GEMV 10.25
# add_definitions(-DMNN_VULKAN_INT4_INDEP)  # tested: indep thread-per-output tg3.69, worst (strided + serial K)
# EXPERIMENT(result: tg 9.9->11.3, barrier ~0.07ms each, NOT the main cost):
# add_definitions(-DMNN_VULKAN_DISABLE_INPUT_BARRIER)

add_subdirectory(${MNN_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/mnn)

include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include
                    ${MNN_ROOT}/include
                    ${MNN_ROOT}/source
                    ${MNN_ROOT}/schema/current
                    ${MNN_ROOT}/3rd_party
                    ${MNN_ROOT}/3rd_party/flatbuffers/include
                    ${MNN_ROOT}/3rd_party/half
                    ${MNN_ROOT}/transformers/llm/engine/include
                    ${MNN_ROOT}/transformers/diffusion/engine/include)

add_definitions(-Wno-format-security)
# llm_napi.cpp embeds an fp16 NEON FMA microbenchmark (inline asm .8h form);
# the assembler needs the fp16 feature. Harmless for the rest of the TU: all
# aarch64 HarmonyOS devices implement ARMv8.2+FP16.
set_source_files_properties(llm_napi.cpp PROPERTIES COMPILE_FLAGS "-O2 -march=armv8.2-a+fp16")
# gemvbw_roofline.cpp uses int8 sdot inline asm (compute roofline microbench).
set_source_files_properties(gemvbw_roofline.cpp PROPERTIES COMPILE_FLAGS "-O2 -march=armv8.2-a+fp16+dotprod")
# llm_bench.cpp is compiled in so its llm_bench_main can be driven through the
# runNativeBench NAPI entry: HarmonyOS NEXT code signing forbids executing
# standalone binaries from /data, so the app process is the CLI test channel.
add_library(entry SHARED test.cpp llm_session.cpp llm_napi.cpp diffusion_session.cpp
            gemvbw_roofline.cpp
            ${MNN_ROOT}/transformers/llm/engine/tools/llm_bench.cpp)
target_link_libraries(entry PUBLIC MNN libace_napi.z.so libhilog_ndk.z.so)

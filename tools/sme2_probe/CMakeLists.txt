# Standalone SME2 hardware probe for Apple Silicon; no MNN dependency.
if(APPLE AND (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"))
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../)
    add_executable(MNNSme2Probe ${CMAKE_CURRENT_LIST_DIR}/sme2_probe.c)
    target_compile_options(MNNSme2Probe PRIVATE -O2)
endif()

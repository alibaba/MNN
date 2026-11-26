target_sources(llm PRIVATE "${CMAKE_CURRENT_LIST_DIR}/LlmDelegate.cpp")
set_property(SOURCE "${CMAKE_CURRENT_LIST_DIR}/../src/llm.cpp"
             APPEND PROPERTY COMPILE_DEFINITIONS MNN_LLM_DELEGATES)
include("${CMAKE_CURRENT_LIST_DIR}/torq/torq.cmake")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/PluginMatMulImpl.cpp
  COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_CURRENT_LIST_DIR}/PluginMatMulImpl.cpp.s
      ${CMAKE_CURRENT_BINARY_DIR}/PluginMatMulImpl.cpp
)

if (MNN_BUILD_SHARED_LIBS)
  add_library(plugin_matmul SHARED ${CMAKE_CURRENT_BINARY_DIR}/PluginMatMulImpl.cpp)
else()
  add_library(plugin_matmul STATIC ${CMAKE_CURRENT_BINARY_DIR}/PluginMatMulImpl.cpp)
endif()
target_link_libraries(plugin_matmul PRIVATE MNN)
target_include_directories(plugin_matmul PRIVATE ${CMAKE_CURRENT_LIST_DIR})

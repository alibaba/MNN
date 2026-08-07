set(ArgumentParser_EXPORTS_FILE ${CMAKE_CURRENT_BINARY_DIR}/ArgumentParserExports.cmake)

 configure_file(ArgumentParserConfig.cmake.in
   ${CMAKE_CURRENT_BINARY_DIR}/ArgumentParserConfig.cmake)

 get_property(ArgumentParser_EXPORTS GLOBAL PROPERTY ArgumentParser_EXPORTS)
 export(TARGETS ${ArgumentParser_EXPORTS}
   FILE ${ArgumentParser_EXPORTS_FILE})
add_library(libcmark-gfm-extensions
  autolink.c
  core-extensions.c
  ext_scanners.c
  ext_scanners.h
  ext_scanners.re
  strikethrough.c
  table.c
  tagfilter.c
  tasklist.c)
target_compile_definitions(libcmark-gfm-extensions PUBLIC
  $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:CMARK_GFM_STATIC_DEFINE>)
target_include_directories(libcmark-gfm-extensions PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cmark_gfm_extensions>)
target_link_libraries(libcmark-gfm-extensions PRIVATE
  libcmark-gfm)
set_target_properties(libcmark-gfm-extensions PROPERTIES
  DEFINE_SYMBOL libcmark_gfm_EXPORTS
  MACOSX_RPATH TRUE
  OUTPUT_NAME cmark-gfm-extensions
  PDB_NAME libcmark-gfm-extensions
  POSITION_INDEPENDENT_CODE YES
  SOVERSION ${PROJECT_VERSION}
  VERSION ${PROJECT_VERSION})


install(TARGETS libcmark-gfm-extensions
  EXPORT cmark-gfm-extensions
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
  include/cmark-gfm-core-extensions.h
  include/module.modulemap
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cmark_gfm_extensions)
install(EXPORT cmark-gfm-extensions
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cmark-gfm-extensions)

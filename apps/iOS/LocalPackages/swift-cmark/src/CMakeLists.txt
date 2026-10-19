configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/config.h)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libcmark-gfm.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/libcmark-gfm.pc @ONLY)

add_library(libcmark-gfm
  arena.c
  blocks.c
  buffer.c
  cmark.c
  cmark_ctype.c
  commonmark.c
  footnotes.c
  houdini_href_e.c
  houdini_html_e.c
  houdini_html_u.c
  html.c
  inlines.c
  iterator.c
  latex.c
  linked_list.c
  man.c
  map.c
  node.c
  plaintext.c
  plugin.c
  references.c
  registry.c
  render.c
  scanners.c
  scanners.re
  syntax_extension.c
  utf8.c
  xml.c)
if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(libcmark-gfm PUBLIC
    CMARK_GFM_STATIC_DEFINE)
  target_compile_options(libcmark-gfm PUBLIC
    $<$<COMPILE_LANGUAGE:Swift>:-Xcc -DCMARK_GFM_STATIC_DEFINE>)
endif()
target_include_directories(libcmark-gfm PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/extensions/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cmark_gfm>)
  target_link_libraries(libcmark-gfm PRIVATE
    $<$<BOOL:${THREADS_FOUND}>:Threads::Threads>)
set_target_properties(libcmark-gfm PROPERTIES
  MACOSX_RPATH TRUE
  OUTPUT_NAME cmark-gfm
  PDB_NAME libcmark-gfm
  POSITION_INDEPENDENT_CODE YES
  SOVERSION ${PROJECT_VERSION}
  VERSION ${PROJECT_VERSION})

add_executable(cmark-gfm
  ${PROJECT_SOURCE_DIR}/bin/main.c)
target_link_libraries(cmark-gfm
  libcmark-gfm
  libcmark-gfm-extensions)


install(TARGETS cmark-gfm libcmark-gfm
  EXPORT cmark-gfm
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
  include/buffer.h
  include/chunk.h
  include/cmark_ctype.h
  include/cmark-gfm.h
  include/cmark-gfm_config.h
  include/cmark-gfm-extension_api.h
  include/cmark-gfm_version.h
  include/export.h
  include/footnotes.h
  include/houdini.h
  include/html.h
  include/inlines.h
  include/iterator.h
  include/map.h
  include/node.h
  include/parser.h
  include/plugin.h
  include/references.h
  include/registry.h
  include/render.h
  include/scanners.h
  include/syntax_extension.h
  include/utf8.h
  include/module.modulemap
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cmark_gfm)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libcmark-gfm.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
install(EXPORT cmark-gfm
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cmark-gfm)

export(TARGETS libcmark-gfm
  FILE ${CMAKE_CURRENT_BINARY_DIR}/cmarkTargets.cmake)

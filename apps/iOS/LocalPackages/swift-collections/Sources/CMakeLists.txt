#[[
This source file is part of the Swift Collections Open Source Project

Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
Licensed under Apache License v2.0 with Runtime Library Exception

See https://swift.org/LICENSE.txt for license information
#]]

if(COLLECTIONS_SINGLE_MODULE)
  if (COLLECTIONS_FOUNDATION_TOOLCHAIN_MODULE)
    set(COLLECTIONS_MODULE_NAME _FoundationCollections)
    set(COLLECTIONS_MODULE_TYPE STATIC)
  else()
    set(COLLECTIONS_MODULE_NAME Collections)
    set(COLLECTIONS_MODULE_TYPE)
  endif()

  add_library(${COLLECTIONS_MODULE_NAME} ${COLLECTIONS_MODULE_TYPE})

  if(COLLECTIONS_FOUNDATION_TOOLCHAIN_MODULE)
    target_compile_options(${COLLECTIONS_MODULE_NAME} PRIVATE
      "SHELL:$<$<COMPILE_LANGUAGE:Swift>:-Xfrontend -allowable-client -Xfrontend FoundationEssentials>")
  endif()

  target_compile_definitions(${COLLECTIONS_MODULE_NAME} PRIVATE COLLECTIONS_SINGLE_MODULE)
  set_target_properties(${COLLECTIONS_MODULE_NAME} PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})

  if(COLLECTIONS_FOUNDATION_TOOLCHAIN_MODULE)
    get_swift_host_os(swift_os)
    if(BUILD_SHARED_LIBS)
      # Install only the swift module and not the library
      set(swift swift)
    else()
      # Install both the swift module and the binary
      set(swift swift_static)
      install(TARGETS ${COLLECTIONS_MODULE_NAME}
          ARCHIVE DESTINATION lib/swift_static/${swift_os}
          LIBRARY DESTINATION lib/swift_static/${swift_os}
          RUNTIME DESTINATION bin)
    endif()
    install(FILES $<TARGET_PROPERTY:${COLLECTIONS_MODULE_NAME},Swift_MODULE_DIRECTORY>/${COLLECTIONS_MODULE_NAME}.swiftdoc
      DESTINATION lib/${swift}/${swift_os}/${COLLECTIONS_MODULE_NAME}.swiftmodule
      RENAME ${SwiftCollections_MODULE_TRIPLE}.swiftdoc)
    install(FILES $<TARGET_PROPERTY:${COLLECTIONS_MODULE_NAME},Swift_MODULE_DIRECTORY>/${COLLECTIONS_MODULE_NAME}.swiftmodule
      DESTINATION lib/${swift}/${swift_os}/${COLLECTIONS_MODULE_NAME}.swiftmodule
      RENAME ${SwiftCollections_MODULE_TRIPLE}.swiftmodule)
  else()
    _install_target(${COLLECTIONS_MODULE_NAME})
  endif()
  set_property(GLOBAL APPEND PROPERTY SWIFT_COLLECTIONS_EXPORTS ${COLLECTIONS_MODULE_NAME})
else()
  add_subdirectory(Collections)
endif()

if(NOT COLLECTIONS_FOUNDATION_TOOLCHAIN_MODULE)
  add_subdirectory(BitCollections)
  add_subdirectory(HashTreeCollections)
  add_subdirectory(HeapModule)
endif()
add_subdirectory(DequeModule)
add_subdirectory(OrderedCollections)
add_subdirectory(RopeModule)
add_subdirectory(InternalCollectionsUtilities)

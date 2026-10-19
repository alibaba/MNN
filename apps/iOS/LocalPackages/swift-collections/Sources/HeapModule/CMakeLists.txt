#[[
This source file is part of the Swift Collections Open Source Project

Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
Licensed under Apache License v2.0 with Runtime Library Exception

See https://swift.org/LICENSE.txt for license information
#]]

if(COLLECTIONS_SINGLE_MODULE)
  set(module_name ${COLLECTIONS_MODULE_NAME})
else()
  set(module_name HeapModule)
  add_library(HeapModule
    ${COLLECTIONS_HEAP_SOURCES})
  target_link_libraries(HeapModule PRIVATE
    InternalCollectionsUtilities)
  set_target_properties(HeapModule PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})

  _install_target(HeapModule)
  set_property(GLOBAL APPEND PROPERTY SWIFT_COLLECTIONS_EXPORTS HeapModule)
endif()

target_sources(${module_name} PRIVATE
  "_HeapNode.swift"
  "Heap.swift"
  "Heap+Descriptions.swift"
  "Heap+ExpressibleByArrayLiteral.swift"
  "Heap+Invariants.swift"
  "Heap+UnsafeHandle.swift")

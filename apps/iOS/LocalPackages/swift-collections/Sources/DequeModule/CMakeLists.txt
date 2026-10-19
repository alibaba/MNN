#[[
This source file is part of the Swift Collections Open Source Project

Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
Licensed under Apache License v2.0 with Runtime Library Exception

See https://swift.org/LICENSE.txt for license information
#]]

if(COLLECTIONS_SINGLE_MODULE)
  set(module_name ${COLLECTIONS_MODULE_NAME})
else()
  set(module_name DequeModule)
  add_library(DequeModule
    ${COLLECTIONS_DEQUE_SOURCES})
  target_link_libraries(DequeModule PRIVATE
    InternalCollectionsUtilities)
  set_target_properties(DequeModule PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})

  _install_target(DequeModule)
  set_property(GLOBAL APPEND PROPERTY SWIFT_COLLECTIONS_EXPORTS DequeModule)
endif()

target_sources(${module_name} PRIVATE
  "Deque+Codable.swift"
  "Deque+Collection.swift"
  "Deque+CustomReflectable.swift"
  "Deque+Descriptions.swift"
  "Deque+Equatable.swift"
  "Deque+ExpressibleByArrayLiteral.swift"
  "Deque+Extras.swift"
  "Deque+Hashable.swift"
  "Deque+Testing.swift"
  "Deque._Storage.swift"
  "Deque._UnsafeHandle.swift"
  "Deque.swift"
  "_DequeBuffer.swift"
  "_DequeBufferHeader.swift"
  "_DequeSlot.swift"
  "_UnsafeWrappedBuffer.swift"
  )

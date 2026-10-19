#[[
This source file is part of the Swift Collections Open Source Project

Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
Licensed under Apache License v2.0 with Runtime Library Exception

See https://swift.org/LICENSE.txt for license information
#]]

add_library(Collections
    "Collections.swift")
target_link_libraries(Collections PRIVATE
  BitCollections
  DequeModule
  HeapModule
  OrderedCollections
  HashTreeCollections
  )
set_target_properties(Collections PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})

_install_target(Collections)
set_property(GLOBAL APPEND PROPERTY SWIFT_COLLECTIONS_EXPORTS Collections)

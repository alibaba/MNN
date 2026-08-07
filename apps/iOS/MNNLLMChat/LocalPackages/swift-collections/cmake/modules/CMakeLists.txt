#[[
This source file is part of the Swift Collections Open Source Project

Copyright (c) 2021 - 2024 Apple Inc. and the Swift project authors
Licensed under Apache License v2.0 with Runtime Library Exception

See https://swift.org/LICENSE.txt for license information
#]]

set(SWIFT_COLLECTIONS_EXPORTS_FILE ${CMAKE_CURRENT_BINARY_DIR}/SwiftCollectionsExports.cmake)

configure_file(SwiftCollectionsConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/SwiftCollectionsConfig.cmake)

get_property(SWIFT_COLLECTIONS_EXPORTS GLOBAL PROPERTY SWIFT_COLLECTIONS_EXPORTS)
export(TARGETS ${SWIFT_COLLECTIONS_EXPORTS}
  NAMESPACE SwiftCollections::
  FILE ${SWIFT_COLLECTIONS_EXPORTS_FILE}
  EXPORT_LINK_INTERFACE_LIBRARIES)

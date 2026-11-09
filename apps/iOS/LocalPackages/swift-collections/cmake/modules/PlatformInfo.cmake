#[[
This source file is part of the Swift Collections Open Source Project

Copyright (c) 2025 Apple Inc. and the Swift project authors
Licensed under Apache License v2.0 with Runtime Library Exception

See https://swift.org/LICENSE.txt for license information
#]]

set(target_info_cmd "${CMAKE_Swift_COMPILER}" -print-target-info)
if(CMAKE_Swfit_COMPILER_TARGET)
  list(APPEND target_info_cmd -target ${CMAKE_Swift_COMPILER_TARGET})
endif()
execute_process(COMMAND ${target_info_cmd} OUTPUT_VARIABLE target_info_json)
message(CONFIGURE_LOG "Swift target info: ${target_info_cmd}\n"
"${target_info_json}")

if(NOT SwiftCollections_MODULE_TRIPLE)
  string(JSON module_triple GET "${target_info_json}" "target" "moduleTriple")
  set(SwiftCollections_MODULE_TRIPLE "${module_triple}" CACHE STRING "Triple used for installed swift{doc,module, interface} files")
  mark_as_advanced(SwiftCollections_MODULE_TRIPLE)

  message(CONFIGURE_LOG "Swift module triple: ${module_triple}")
endif()

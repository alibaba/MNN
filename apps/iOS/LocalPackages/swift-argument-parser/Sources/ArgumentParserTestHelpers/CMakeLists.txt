add_library(ArgumentParserTestHelpers
  StringHelpers.swift
  TestHelpers.swift)
set_target_properties(ArgumentParserTestHelpers PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_link_libraries(ArgumentParserTestHelpers PUBLIC
  ArgumentParser
  ArgumentParserToolInfo
  XCTest
  Foundation)

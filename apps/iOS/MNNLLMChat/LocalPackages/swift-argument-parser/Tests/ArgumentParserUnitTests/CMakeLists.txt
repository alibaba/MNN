add_library(UnitTests
  ParsableArgumentsValidationTests.swift
  ErrorMessageTests.swift
  HelpGenerationTests.swift
  HelpGenerationTests+AtArgument.swift
  HelpGenerationTests+AtOption.swift
  HelpGenerationTests+GroupName.swift
  NameSpecificationTests.swift
  SplitArgumentTests.swift
  StringSnakeCaseTests.swift
  StringWrappingTests.swift
  TreeTests.swift
  UsageGenerationTests.swift)
target_link_libraries(UnitTests PRIVATE
  ArgumentParser
  ArgumentParserTestHelpers)

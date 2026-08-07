add_library(ArgumentParser
  Completions/BashCompletionsGenerator.swift
  Completions/CompletionsGenerator.swift
  Completions/FishCompletionsGenerator.swift
  Completions/ZshCompletionsGenerator.swift

  "Parsable Properties/Argument.swift"
  "Parsable Properties/ArgumentHelp.swift"
  "Parsable Properties/ArgumentVisibility.swift"
  "Parsable Properties/CompletionKind.swift"
  "Parsable Properties/Errors.swift"
  "Parsable Properties/Flag.swift"
  "Parsable Properties/NameSpecification.swift"
  "Parsable Properties/Option.swift"
  "Parsable Properties/OptionGroup.swift"

  "Parsable Types/AsyncParsableCommand.swift"
  "Parsable Types/CommandConfiguration.swift"
  "Parsable Types/EnumerableFlag.swift"
  "Parsable Types/ExpressibleByArgument.swift"
  "Parsable Types/ParsableArguments.swift"
  "Parsable Types/ParsableArgumentsValidation.swift"
  "Parsable Types/ParsableCommand.swift"

  Parsing/ArgumentDecoder.swift
  Parsing/ArgumentDefinition.swift
  Parsing/ArgumentSet.swift
  Parsing/CommandParser.swift
  Parsing/InputKey.swift
  Parsing/InputOrigin.swift
  Parsing/Name.swift
  Parsing/Parsed.swift
  Parsing/ParsedValues.swift
  Parsing/ParserError.swift
  Parsing/SplitArguments.swift

  Usage/DumpHelpGenerator.swift
  Usage/HelpCommand.swift
  Usage/HelpGenerator.swift
  Usage/MessageInfo.swift
  Usage/UsageGenerator.swift

  Utilities/CollectionExtensions.swift
  Utilities/Platform.swift
  Utilities/SequenceExtensions.swift
  Utilities/StringExtensions.swift
  Utilities/Tree.swift)
# NOTE: workaround for CMake not setting up include flags yet
set_target_properties(ArgumentParser PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY})
target_compile_options(ArgumentParser PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)
target_link_libraries(ArgumentParser PRIVATE
  $<$<NOT:$<PLATFORM_ID:Darwin>>:Foundation>
  ArgumentParserToolInfo)


_install_target(ArgumentParser)
set_property(GLOBAL APPEND PROPERTY ArgumentParser_EXPORTS ArgumentParser)

//===----------------------------------------------------------*- swift -*-===//
//
// This source file is part of the Swift Argument Parser open source project
//
// Copyright (c) 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

struct ZshCompletionsGenerator {
  /// Generates a Zsh completion script for the given command.
  static func generateCompletionScript(_ type: ParsableCommand.Type) -> String {
    let initialFunctionName = [type].completionFunctionName()

    return """
    #compdef \(type._commandName)
    local context state state_descr line
    _\(type._commandName.zshEscapingCommandName())_commandname=$words[1]
    typeset -A opt_args

    \(generateCompletionFunction([type]))
    _custom_completion() {
        local completions=("${(@f)$($*)}")
        _describe '' completions
    }

    \(initialFunctionName)
    """
  }
  
  static func generateCompletionFunction(_ commands: [ParsableCommand.Type]) -> String {
    let type = commands.last!
    let functionName = commands.completionFunctionName()
    let isRootCommand = commands.count == 1
    
    var args = generateCompletionArguments(commands)    
    var subcommands = type.configuration.subcommands
      .filter { $0.configuration.shouldDisplay }
    var subcommandHandler = ""
    if !subcommands.isEmpty {
      args.append("'(-): :->command'")
      args.append("'(-)*:: :->arg'")
      
      if isRootCommand {
        subcommands.append(HelpCommand.self)
      }

      let subcommandModes = subcommands.map {
        """
        '\($0._commandName):\($0.configuration.abstract.zshEscaped())'
        """
        .indentingEachLine(by: 12)
      }
      let subcommandArgs = subcommands.map {
        """
        (\($0._commandName))
            \(functionName)_\($0._commandName)
            ;;
        """
        .indentingEachLine(by: 12)
      }
      
      subcommandHandler = """
        case $state in
            (command)
                local subcommands
                subcommands=(
        \(subcommandModes.joined(separator: "\n"))
                )
                _describe "subcommand" subcommands
                ;;
            (arg)
                case ${words[1]} in
        \(subcommandArgs.joined(separator: "\n"))
                esac
                ;;
        esac
        
        """
        .indentingEachLine(by: 4)
    }
    
    let functionText = """
      \(functionName)() {
          integer ret=1
          local -a args
          args+=(
      \(args.joined(separator: "\n").indentingEachLine(by: 8))
          )
          _arguments -w -s -S $args[@] && ret=0
      \(subcommandHandler)
          return ret
      }
      
      
      """
    
    return functionText +
      subcommands
        .map { generateCompletionFunction(commands + [$0]) }
        .joined()
  }

  static func generateCompletionArguments(_ commands: [ParsableCommand.Type]) -> [String] {
    commands
      .argumentsForHelp(visibility: .default)
      .compactMap { $0.zshCompletionString(commands) }
  }
}

extension String {
  fileprivate func zshEscapingSingleQuotes() -> String {
    self.replacingOccurrences(of: "'", with: #"'"'"'"#)
  }

  fileprivate func zshEscapingMetacharacters() -> String {
    self.replacingOccurrences(of: #"[\\\[\]]"#, with: #"\\$0"#, options: .regularExpression)
  }

  fileprivate func zshEscaped() -> String {
    self.zshEscapingSingleQuotes().zshEscapingMetacharacters()
  }
  
  fileprivate func zshEscapingCommandName() -> String {
    self.replacingOccurrences(of: "-", with: "_")
  }
}

extension ArgumentDefinition {
  var zshCompletionAbstract: String {
    guard !help.abstract.isEmpty else { return "" }
    return "[\(help.abstract.zshEscaped())]"
  }
  
  func zshCompletionString(_ commands: [ParsableCommand.Type]) -> String? {
    guard help.visibility.base == .default else { return nil }
    
    let inputs: String
    switch update {
    case .unary:
      inputs = ":\(valueName):\(zshActionString(commands))"
    case .nullary:
      inputs = ""
    }

    let line: String
    switch names.count {
    case 0:
      line = ""
    case 1:
      let star = isRepeatableOption ? "*" : ""
      line = """
      \(star)\(names[0].synopsisString)\(zshCompletionAbstract)
      """
    default:
      let synopses = names.map { $0.synopsisString }
      let suppression = isRepeatableOption ? "*" : "(\(synopses.joined(separator: " ")))"
      line = """
      \(suppression)'\
      {\(synopses.joined(separator: ","))}\
      '\(zshCompletionAbstract)
      """
    }
    
    return "'\(line)\(inputs)'"
  }

  /// - returns: `true` if I'm an option and can be tab-completed multiple times in one command line. For example, `ssh` allows the `-L` option to be given multiple times, to establish multiple port forwardings.
  private var isRepeatableOption: Bool {
    guard
      case .named(_) = kind,
      help.options.contains(.isRepeating)
    else { return false }

    switch parsingStrategy {
    case .default, .unconditional: return true
    default: return false
    }
  }

  /// Returns the zsh "action" for an argument completion string.
  func zshActionString(_ commands: [ParsableCommand.Type]) -> String {
    switch completion.kind {
    case .default:
      return ""
      
    case .file(let extensions):
      let pattern = extensions.isEmpty
        ? ""
        : " -g '\(extensions.map { "*." + $0 }.joined(separator: " "))'"
      return "_files\(pattern.zshEscaped())"
      
    case .directory:
      return "_files -/"
      
    case .list(let list):
      return "(" + list.joined(separator: " ") + ")"
      
    case .shellCommand(let command):
      return "{local -a list; list=(${(f)\"$(\(command))\"}); _describe '''' list}"

    case .custom:
      // Generate a call back into the command to retrieve a completions list
      let commandName = commands.first!._commandName.zshEscapingCommandName()
      return "{_custom_completion $_\(commandName)_commandname \(customCompletionCall(commands)) $words}"
    }
  }
}


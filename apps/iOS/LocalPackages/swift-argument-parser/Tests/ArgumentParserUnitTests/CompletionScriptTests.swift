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

import XCTest
import ArgumentParserTestHelpers
@testable import ArgumentParser

final class CompletionScriptTests: XCTestCase {
}

extension CompletionScriptTests {
  struct Path: ExpressibleByArgument {
    var path: String
    
    init?(argument: String) {
      self.path = argument
    }
    
    static var defaultCompletionKind: CompletionKind {
      .file()
    }
  }
    
  enum Kind: String, ExpressibleByArgument, EnumerableFlag {
    case one, two, three = "custom-three"
  }
  
  struct Base: ParsableCommand {
    static let configuration = CommandConfiguration(
      commandName: "base-test",
      subcommands: [SubCommand.self]
    )

    @Option(help: "The user's name.") var name: String
    @Option() var kind: Kind
    @Option(completion: .list(["1", "2", "3"])) var otherKind: Kind
    
    @Option() var path1: Path
    @Option() var path2: Path?
    @Option(completion: .list(["a", "b", "c"])) var path3: Path
    
    @Flag(help: .hidden) var verbose = false
    @Flag var allowedKinds: [Kind] = []
    @Flag var kindCounter: Int
    
    @Option() var rep1: [String]
    @Option(name: [.short, .long]) var rep2: [String]

   struct SubCommand: ParsableCommand {
     static let configuration = CommandConfiguration(
       commandName: "sub-command"
     )
   }
  }

  func testBase_Zsh() throws {
    let script1 = try CompletionsGenerator(command: Base.self, shell: .zsh)
          .generateCompletionScript()
    XCTAssertEqual(zshBaseCompletions, script1)
    
    let script2 = try CompletionsGenerator(command: Base.self, shellName: "zsh")
          .generateCompletionScript()
    XCTAssertEqual(zshBaseCompletions, script2)
    
    let script3 = Base.completionScript(for: .zsh)
    XCTAssertEqual(zshBaseCompletions, script3)
  }

  func testBase_Bash() throws {
    let script1 = try CompletionsGenerator(command: Base.self, shell: .bash)
          .generateCompletionScript()
    XCTAssertEqual(bashBaseCompletions, script1)
    
    let script2 = try CompletionsGenerator(command: Base.self, shellName: "bash")
          .generateCompletionScript()
    XCTAssertEqual(bashBaseCompletions, script2)
    
    let script3 = Base.completionScript(for: .bash)
    XCTAssertEqual(bashBaseCompletions, script3)
  }

  func testBase_Fish() throws {
    let script1 = try CompletionsGenerator(command: Base.self, shell: .fish)
          .generateCompletionScript()
    XCTAssertEqual(fishBaseCompletions, script1)
    
    let script2 = try CompletionsGenerator(command: Base.self, shellName: "fish")
          .generateCompletionScript()
    XCTAssertEqual(fishBaseCompletions, script2)
    
    let script3 = Base.completionScript(for: .fish)
    XCTAssertEqual(fishBaseCompletions, script3)
  }
}

extension CompletionScriptTests {
  struct Custom: ParsableCommand {
    @Option(name: .shortAndLong, completion: .custom { _ in ["a", "b", "c"] })
    var one: String

    @Argument(completion: .custom { _ in ["d", "e", "f"] })
    var two: String

    @Option(name: .customShort("z"), completion: .custom { _ in ["x", "y", "z"] })
    var three: String
  }
  
  func verifyCustomOutput(
    _ arg: String,
    expectedOutput: String,
    file: StaticString = #file, line: UInt = #line
  ) throws {
    do {
      _ = try Custom.parse(["---completion", "--", arg])
      XCTFail("Didn't error as expected", file: (file), line: line)
    } catch let error as CommandError {
      guard case .completionScriptCustomResponse(let output) = error.parserError else {
        throw error
      }
      XCTAssertEqual(expectedOutput, output, file: (file), line: line)
    }
  }
  
  func testCustomCompletions() throws {
    try verifyCustomOutput("-o", expectedOutput: "a\nb\nc")
    try verifyCustomOutput("--one", expectedOutput: "a\nb\nc")
    try verifyCustomOutput("two", expectedOutput: "d\ne\nf")
    try verifyCustomOutput("-z", expectedOutput: "x\ny\nz")
    
    XCTAssertThrowsError(try verifyCustomOutput("--bad", expectedOutput: ""))
  }
}

extension CompletionScriptTests {
  struct EscapedCommand: ParsableCommand {
    @Option(help: #"Escaped chars: '[]\."#)
    var one: String
    
    @Argument(completion: .custom { _ in ["d", "e", "f"] })
    var two: String
  }

  func testEscaped_Zsh() throws {
    XCTAssertEqual(zshEscapedCompletion, EscapedCommand.completionScript(for: .zsh))
  }
}

private let zshBaseCompletions = """
#compdef base-test
local context state state_descr line
_base_test_commandname=$words[1]
typeset -A opt_args

_base-test() {
    integer ret=1
    local -a args
    args+=(
        '--name[The user'"'"'s name.]:name:'
        '--kind:kind:(one two custom-three)'
        '--other-kind:other-kind:(1 2 3)'
        '--path1:path1:_files'
        '--path2:path2:_files'
        '--path3:path3:(a b c)'
        '--one'
        '--two'
        '--three'
        '*--kind-counter'
        '*--rep1:rep1:'
        '*'{-r,--rep2}':rep2:'
        '(-h --help)'{-h,--help}'[Show help information.]'
        '(-): :->command'
        '(-)*:: :->arg'
    )
    _arguments -w -s -S $args[@] && ret=0
    case $state in
        (command)
            local subcommands
            subcommands=(
                'sub-command:'
                'help:Show subcommand help information.'
            )
            _describe "subcommand" subcommands
            ;;
        (arg)
            case ${words[1]} in
                (sub-command)
                    _base-test_sub-command
                    ;;
                (help)
                    _base-test_help
                    ;;
            esac
            ;;
    esac

    return ret
}

_base-test_sub-command() {
    integer ret=1
    local -a args
    args+=(
        '(-h --help)'{-h,--help}'[Show help information.]'
    )
    _arguments -w -s -S $args[@] && ret=0

    return ret
}

_base-test_help() {
    integer ret=1
    local -a args
    args+=(
        ':subcommands:'
    )
    _arguments -w -s -S $args[@] && ret=0

    return ret
}


_custom_completion() {
    local completions=("${(@f)$($*)}")
    _describe '' completions
}

_base-test
"""

private let bashBaseCompletions = """
#!/bin/bash

_base_test() {
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    COMPREPLY=()
    opts="--name --kind --other-kind --path1 --path2 --path3 --one --two --three --kind-counter --rep1 -r --rep2 -h --help sub-command help"
    if [[ $COMP_CWORD == "1" ]]; then
        COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
        return
    fi
    case $prev in
        --name)

            return
        ;;
        --kind)
            COMPREPLY=( $(compgen -W "one two custom-three" -- "$cur") )
            return
        ;;
        --other-kind)
            COMPREPLY=( $(compgen -W "1 2 3" -- "$cur") )
            return
        ;;
        --path1)
            if declare -F _filedir >/dev/null; then
              _filedir
            else
              COMPREPLY=( $(compgen -f -- "$cur") )
            fi
            return
        ;;
        --path2)
            if declare -F _filedir >/dev/null; then
              _filedir
            else
              COMPREPLY=( $(compgen -f -- "$cur") )
            fi
            return
        ;;
        --path3)
            COMPREPLY=( $(compgen -W "a b c" -- "$cur") )
            return
        ;;
        --rep1)

            return
        ;;
        -r|--rep2)

            return
        ;;
    esac
    case ${COMP_WORDS[1]} in
        (sub-command)
            _base_test_sub-command 2
            return
            ;;
        (help)
            _base_test_help 2
            return
            ;;
    esac
    COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
}
_base_test_sub_command() {
    opts="-h --help"
    if [[ $COMP_CWORD == "$1" ]]; then
        COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
        return
    fi
    COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
}
_base_test_help() {
    opts=""
    if [[ $COMP_CWORD == "$1" ]]; then
        COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
        return
    fi
    COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
}


complete -F _base_test base-test
"""

private let zshEscapedCompletion = """
#compdef escaped-command
local context state state_descr line
_escaped_command_commandname=$words[1]
typeset -A opt_args

_escaped-command() {
    integer ret=1
    local -a args
    args+=(
        '--one[Escaped chars: '"'"'\\[\\]\\\\.]:one:'
        ':two:{_custom_completion $_escaped_command_commandname ---completion  -- two $words}'
        '(-h --help)'{-h,--help}'[Show help information.]'
    )
    _arguments -w -s -S $args[@] && ret=0

    return ret
}


_custom_completion() {
    local completions=("${(@f)$($*)}")
    _describe '' completions
}

_escaped-command
"""

private let fishBaseCompletions = """
# A function which filters options which starts with "-" from $argv.
function _swift_base-test_preprocessor
    set -l results
    for i in (seq (count $argv))
        switch (echo $argv[$i] | string sub -l 1)
            case '-'
            case '*'
                echo $argv[$i]
        end
    end
end

function _swift_base-test_using_command
    set -l currentCommands (_swift_base-test_preprocessor (commandline -opc))
    set -l expectedCommands (string split " " $argv[1])
    set -l subcommands (string split " " $argv[2])
    if [ (count $currentCommands) -ge (count $expectedCommands) ]
        for i in (seq (count $expectedCommands))
            if [ $currentCommands[$i] != $expectedCommands[$i] ]
                return 1
            end
        end
        if [ (count $currentCommands) -eq (count $expectedCommands) ]
            return 0
        end
        if [ (count $subcommands) -gt 1 ]
            for i in (seq (count $subcommands))
                if [ $currentCommands[(math (count $expectedCommands) + 1)] = $subcommands[$i] ]
                    return 1
                end
            end
        end
        return 0
    end
    return 1
end

complete -c base-test -n '_swift_base-test_using_command "base-test sub-command"' -s h -l help -d 'Show help information.'
complete -c base-test -n '_swift_base-test_using_command "base-test" "sub-command help"' -l name -d 'The user\\'s name.'
complete -c base-test -n '_swift_base-test_using_command "base-test" "sub-command help"' -l kind -r -f -k -a 'one two custom-three'
complete -c base-test -n '_swift_base-test_using_command "base-test" "sub-command help"' -l other-kind -r -f -k -a '1 2 3'
complete -c base-test -n '_swift_base-test_using_command "base-test" "sub-command help"' -l path1 -r -f -a '(for i in *.{}; echo $i;end)'
complete -c base-test -n '_swift_base-test_using_command "base-test" "sub-command help"' -l path2 -r -f -a '(for i in *.{}; echo $i;end)'
complete -c base-test -n '_swift_base-test_using_command "base-test" "sub-command help"' -l path3 -r -f -k -a 'a b c'
complete -c base-test -n '_swift_base-test_using_command "base-test" "sub-command help"' -l one
complete -c base-test -n '_swift_base-test_using_command "base-test" "sub-command help"' -l two
complete -c base-test -n '_swift_base-test_using_command "base-test" "sub-command help"' -l three
complete -c base-test -n '_swift_base-test_using_command "base-test" "sub-command help"' -l kind-counter
complete -c base-test -n '_swift_base-test_using_command "base-test" "sub-command help"' -l rep1
complete -c base-test -n '_swift_base-test_using_command "base-test" "sub-command help"' -s r -l rep2
complete -c base-test -n '_swift_base-test_using_command "base-test" "sub-command help"' -s h -l help -d 'Show help information.'
complete -c base-test -n '_swift_base-test_using_command "base-test" "sub-command help"' -f -a 'sub-command' -d ''
complete -c base-test -n '_swift_base-test_using_command "base-test" "sub-command help"' -f -a 'help' -d 'Show subcommand help information.'
"""

// MARK: - Test Hidden Subcommand
struct Parent: ParsableCommand {
    static let configuration = CommandConfiguration(subcommands: [HiddenChild.self])
}

struct HiddenChild: ParsableCommand {
    static let configuration = CommandConfiguration(shouldDisplay: false)
}

extension CompletionScriptTests {
  func testHiddenSubcommand_Zsh() throws {
    let script1 = try CompletionsGenerator(command: Parent.self, shell: .zsh)
          .generateCompletionScript()
    XCTAssertEqual(zshHiddenCompletion, script1)

    let script2 = try CompletionsGenerator(command: Parent.self, shellName: "zsh")
          .generateCompletionScript()
    XCTAssertEqual(zshHiddenCompletion, script2)

    let script3 = Parent.completionScript(for: .zsh)
    XCTAssertEqual(zshHiddenCompletion, script3)
  }

  func testHiddenSubcommand_Bash() throws {
    let script1 = try CompletionsGenerator(command: Parent.self, shell: .bash)
          .generateCompletionScript()
    XCTAssertEqual(bashHiddenCompletion, script1)

    let script2 = try CompletionsGenerator(command: Parent.self, shellName: "bash")
          .generateCompletionScript()
    XCTAssertEqual(bashHiddenCompletion, script2)

    let script3 = Parent.completionScript(for: .bash)
    XCTAssertEqual(bashHiddenCompletion, script3)
  }

  func testHiddenSubcommand_Fish() throws {
    let script1 = try CompletionsGenerator(command: Parent.self, shell: .fish)
          .generateCompletionScript()
    XCTAssertEqual(fishHiddenCompletion, script1)

    let script2 = try CompletionsGenerator(command: Parent.self, shellName: "fish")
          .generateCompletionScript()
    XCTAssertEqual(fishHiddenCompletion, script2)

    let script3 = Parent.completionScript(for: .fish)
    XCTAssertEqual(fishHiddenCompletion, script3)
  }
}

let zshHiddenCompletion = """
#compdef parent
local context state state_descr line
_parent_commandname=$words[1]
typeset -A opt_args

_parent() {
    integer ret=1
    local -a args
    args+=(
        '(-h --help)'{-h,--help}'[Show help information.]'
    )
    _arguments -w -s -S $args[@] && ret=0

    return ret
}


_custom_completion() {
    local completions=("${(@f)$($*)}")
    _describe '' completions
}

_parent
"""

let bashHiddenCompletion = """
#!/bin/bash

_parent() {
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    COMPREPLY=()
    opts="-h --help"
    if [[ $COMP_CWORD == "1" ]]; then
        COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
        return
    fi
    COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
}


complete -F _parent parent
"""

let fishHiddenCompletion = """
# A function which filters options which starts with "-" from $argv.
function _swift_parent_preprocessor
    set -l results
    for i in (seq (count $argv))
        switch (echo $argv[$i] | string sub -l 1)
            case '-'
            case '*'
                echo $argv[$i]
        end
    end
end

function _swift_parent_using_command
    set -l currentCommands (_swift_parent_preprocessor (commandline -opc))
    set -l expectedCommands (string split " " $argv[1])
    set -l subcommands (string split " " $argv[2])
    if [ (count $currentCommands) -ge (count $expectedCommands) ]
        for i in (seq (count $expectedCommands))
            if [ $currentCommands[$i] != $expectedCommands[$i] ]
                return 1
            end
        end
        if [ (count $currentCommands) -eq (count $expectedCommands) ]
            return 0
        end
        if [ (count $subcommands) -gt 1 ]
            for i in (seq (count $subcommands))
                if [ $currentCommands[(math (count $expectedCommands) + 1)] = $subcommands[$i] ]
                    return 1
                end
            end
        end
        return 0
    end
    return 1
end

complete -c parent -n '_swift_parent_using_command \"parent\"' -s h -l help -d 'Show help information.'
"""

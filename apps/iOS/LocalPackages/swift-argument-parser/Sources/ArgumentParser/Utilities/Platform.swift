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

extension CommandLine {
  /// Accesses the command line arguments in a concurrency-safe way.
  ///
  /// Workaround for https://github.com/apple/swift/issues/66213
  static let _staticArguments: [String] =
    UnsafeBufferPointer(start: unsafeArgv, count: Int(argc))
      .compactMap { String(validatingUTF8: $0!)
  }
}

#if canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#elseif canImport(Darwin)
import Darwin
#elseif canImport(CRT)
import CRT
#elseif canImport(WASILibc)
import WASILibc
#endif

enum Platform {}

// MARK: Shell

extension Platform {
  /// The name of the user's preferred shell, if detectable from the
  /// environment.
  static var shellName: String? {
#if os(Windows)
    return nil
#else
    // FIXME: This retrieves the user's preferred shell, not necessarily the one currently in use.
    guard let shellVar = getenv("SHELL") else { return nil }
    let shellParts = String(cString: shellVar).split(separator: "/")
    return shellParts.last.map(String.init)
#endif
  }
}

// MARK: Exit codes

#if os(Windows)
import func WinSDK.GetStdHandle
import func WinSDK.GetConsoleScreenBufferInfo
import let WinSDK.ERROR_BAD_ARGUMENTS
import let WinSDK.STD_OUTPUT_HANDLE
import struct WinSDK.CONSOLE_SCREEN_BUFFER_INFO
#endif

extension Platform {
  /// The code for successful exit.
  static var exitCodeSuccess: Int32 {
    EXIT_SUCCESS
  }
  
  /// The code for exit with a general failure.
  static var exitCodeFailure: Int32 {
    EXIT_FAILURE
  }
  
  /// The code for exit with a validation failure.
  static var exitCodeValidationFailure: Int32 {
#if os(Windows)
    return ERROR_BAD_ARGUMENTS
#elseif os(WASI)
    return EXIT_FAILURE
#else
    return EX_USAGE
#endif
  }
}

// MARK: Exit function

extension Platform {
  /// Complete execution with the given exit code.
  static func exit(_ code: Int32) -> Never {
#if canImport(Glibc)
    Glibc.exit(code)
#elseif canImport(Musl)
    Musl.exit(code)
#elseif canImport(Darwin)
    Darwin.exit(code)
#elseif canImport(CRT)
    ucrt._exit(code)
#elseif canImport(WASILibc)
    WASILibc.exit(code)
#endif
  }
}

// MARK: Standard error

extension Platform {
  /// A type that represents the `stderr` output stream.
  struct StandardError: TextOutputStream {
    mutating func write(_ string: String) {
      for byte in string.utf8 { putc(numericCast(byte), stderr) }
    }
  }

  /// The `stderr` output stream.
  static var standardError: StandardError {
    StandardError()
  }
}

// MARK: Terminal size

#if canImport(Glibc)
func ioctl(_ a: Int32, _ b: Int32, _ p: UnsafeMutableRawPointer) -> Int32 {
  ioctl(CInt(a), UInt(b), p)
}
#endif

extension Platform {
  /// The default terminal size.
  private static var defaultTerminalSize: (width: Int, height: Int) {
    (width: 80, height: 25)
  }
  
  /// The terminal size specified by the COLUMNS and LINES overrides
  /// (if present).
  ///
  /// Per the [Linux environ(7) manpage][linenv]:
  ///
  /// ```
  /// * COLUMNS and LINES tell applications about the window size,
  ///   possibly overriding the actual size.
  /// ```
  ///
  /// And the [FreeBSD environ(7) version][bsdenv]:
  ///
  /// ```
  /// COLUMNS    The user's preferred width in column positions for the
  ///            terminal.  Utilities such as ls(1) and who(1) use this
  ///            to format output into columns.   If  unset  or  empty,
  ///            utilities  will use an ioctl(2) call to ask the termi-
  ///            nal driver for the width.
  /// ```
  ///
  /// > Note: Always returns `(nil, nil)` on Windows and WASI.
  ///
  /// - Returns: A tuple consisting of a width found in the `COLUMNS` environment
  ///   variable (or `nil` if the variable is not present) and a height found in
  ///   the `LINES` environment variable (or `nil` if that variable is not present).
  ///
  /// [linenv]: https://man7.org/linux/man-pages/man7/environ.7.html:~:text=COLUMNS
  /// [bsdenv]: https://man.freebsd.org/cgi/man.cgi?environ(7)#:~:text=COLUMNS
  private static func userSpecifiedTerminalSize() -> (width: Int?, height: Int?) {
    var width: Int? = nil, height: Int? = nil

#if !os(Windows) && !os(WASI)
    if let colsCStr = getenv("COLUMNS"), let colsVal = Int(String(cString: colsCStr)) {
      width = colsVal
    }
    if let linesCStr = getenv("LINES"), let linesVal = Int(String(cString: linesCStr)) {
      height = linesVal
    }
#endif

    return (width: width, height: height)
  }
  
  /// The current terminal size as reported by the windowing system,
  /// if available.
  ///
  /// Returns (nil, nil) if no reported size is available.
  private static func reportedTerminalSize() -> (width: Int?, height: Int?) {
#if os(WASI)
    // WASI doesn't yet support terminal size
    return (width: nil, height: nil)
#elseif os(Windows)
    var csbi = CONSOLE_SCREEN_BUFFER_INFO()
    guard GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi) else {
      return (width: nil, height: nil)
    }
    return (width: Int(csbi.srWindow.Right - csbi.srWindow.Left) + 1,
            height: Int(csbi.srWindow.Bottom - csbi.srWindow.Top) + 1)
#else
    var w = winsize()

#if os(OpenBSD)
    // TIOCGWINSZ is a complex macro, so we need the flattened value.
    let tiocgwinsz = Int32(0x40087468)
    let err = ioctl(STDOUT_FILENO, tiocgwinsz, &w)
#elseif canImport(Musl)
    let err = ioctl(STDOUT_FILENO, UInt(TIOCGWINSZ), &w)
#else
    let err = ioctl(STDOUT_FILENO, TIOCGWINSZ, &w)
#endif
    guard err == 0 else { return (width: nil, height: nil) }

    let width = Int(w.ws_col), height = Int(w.ws_row)
    
    return (width: width > 0 ? width : nil,
            height: height > 0 ? height : nil)
#endif
  }
  
  /// Returns the current terminal size, or the default if the size is unavailable.
  static func terminalSize() -> (width: Int, height: Int) {
    let specifiedSize = self.userSpecifiedTerminalSize()
    
    // Avoid needlessly calling ioctl() if a complete override is in effect
    if let specifiedWidth = specifiedSize.width, let specifiedHeight = specifiedSize.height {
        return (width: specifiedWidth, height: specifiedHeight)
    }
    
    // Get the size self-reported by the terminal, if available
    let reportedSize = self.reportedTerminalSize()
    
    // As it isn't required that both width and height always be specified
    // together, either by the user or the terminal itself, they are
    // handled separately.
    return (
      width: specifiedSize.width ?? reportedSize.width ?? defaultTerminalSize.width,
      height: specifiedSize.height ?? reportedSize.height ?? defaultTerminalSize.height
    )
  }
  
  /// The current terminal size, or the default if the width is unavailable.
  static var terminalWidth: Int {
    self.terminalSize().width
  }
}

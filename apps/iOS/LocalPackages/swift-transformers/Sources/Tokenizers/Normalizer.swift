//
//  Normalizer.swift
//
//
//  Created by Pedro Cuenca on 17/7/23.
//

import Foundation
import Hub

public protocol Normalizer {
    func normalize(text: String) -> String
    func callAsFunction(text: String) -> String

    init(config: Config)
}

extension Normalizer {
    func callAsFunction(text: String) -> String {
        normalize(text: text)
    }
}

enum NormalizerType: String {
    case Sequence
    case Prepend
    case Replace
    case Lowercase
    case NFD
    case NFC
    case NFKD
    case NFKC
    case Bert
    case BertNormalizer
    case Precompiled
    case StripAccents
    case Strip
    case Unknown = ""
}

struct NormalizerFactory {
    static func fromConfig(config: Config?) -> Normalizer? {
        guard let config else { return nil }
        guard let typeName = config.type.string() else { return nil }
        let type = NormalizerType(rawValue: typeName)
        switch type {
        case .Sequence: return NormalizerSequence(config: config)
        case .Prepend: return PrependNormalizer(config: config)
        case .Replace: return ReplaceNormalizer(config: config)
        case .Lowercase: return LowercaseNormalizer(config: config)
        case .NFD: return NFDNormalizer(config: config)
        case .NFC: return NFCNormalizer(config: config)
        case .NFKD: return NFKDNormalizer(config: config)
        case .NFKC: return NFKCNormalizer(config: config)
        case .Bert, .BertNormalizer: return BertNormalizer(config: config)
        case .Precompiled: return PrecompiledNormalizer(config: config)
        case .StripAccents: return StripAccentsNormalizer(config: config)
        case .Strip: return StripNormalizer(config: config)
        default: fatalError("Unsupported Normalizer type: \(typeName)")
        }
    }
}

class NormalizerSequence: Normalizer {
    let normalizers: [Normalizer]

    public required init(config: Config) {
        guard let configs = config.normalizers.array() else {
            fatalError("No normalizers in Sequence")
        }
        normalizers = configs.compactMap { NormalizerFactory.fromConfig(config: $0) }
    }

    public func normalize(text: String) -> String {
        normalizers.reduce(text) { current, normalizer in
            normalizer(text: current)
        }
    }
}

class PrependNormalizer: Normalizer {
    let prepend: String

    public required init(config: Config) {
        prepend = config.prepend.string(or: "")
    }

    public func normalize(text: String) -> String {
        prepend + text
    }
}

class ReplaceNormalizer: Normalizer {
    let pattern: StringReplacePattern?

    public required init(config: Config) {
        pattern = StringReplacePattern.from(config: config)
    }

    public func normalize(text: String) -> String {
        guard let pattern else { return text }
        return pattern.replace(text)
    }
}

class LowercaseNormalizer: Normalizer {
    public required init(config: Config) { }

    public func normalize(text: String) -> String {
        text.lowercased()
    }
}

class NFDNormalizer: Normalizer {
    public required init(config: Config) { }

    public func normalize(text: String) -> String {
        text.decomposedStringWithCanonicalMapping
    }
}

class NFCNormalizer: Normalizer {
    public required init(config: Config) { }

    public func normalize(text: String) -> String {
        text.precomposedStringWithCanonicalMapping
    }
}

class NFKDNormalizer: Normalizer {
    required init(config: Config) { }

    func normalize(text: String) -> String {
        text.decomposedStringWithCompatibilityMapping
    }
}

class NFKCNormalizer: Normalizer {
    required init(config: Config) { }

    func normalize(text: String) -> String {
        text.precomposedStringWithCompatibilityMapping
    }
}

class BertNormalizer: Normalizer {
    let shouldCleanText: Bool
    let shouldHandleChineseChars: Bool
    let shouldStripAccents: Bool
    let shouldLowercase: Bool

    required init(config: Config) {
        shouldCleanText = config.cleanText.boolean(or: true)
        shouldHandleChineseChars = config.handleChineseChars.boolean(or: true)
        shouldLowercase = config.lowercase.boolean(or: true)
        shouldStripAccents = config.stripAccents.boolean(or: shouldLowercase)
    }

    func normalize(text: String) -> String {
        var output = text
        if shouldCleanText {
            output = cleanText(text: output)
        }
        if shouldHandleChineseChars {
            output = handleChineseChars(text: output)
        }
        if shouldStripAccents {
            output = stripAccents(text: output)
        }
        if shouldLowercase {
            output = output.lowercased()
        }

        return output
    }

    private func cleanText(text: String) -> String {
        text.map { c in
            guard let scalar = c.unicodeScalars.first,
                  scalar.value != 0x0,
                  scalar.value != 0xFFFD,
                  !isControl(scalar)
            else { return "\(c)" }

            // Replace whitespace: \t, \n, \r
            if scalar.value == 0x009 || scalar.value == 0x00A || scalar.value == 0x000D {
                return " "
            } else {
                return "\(c)"
            }
        }
        .joined()
    }

    private func isControl(_ c: UnicodeScalar) -> Bool {
        if c.value == 0x009 || c.value == 0x00A || c.value == 0x000D {
            // Except \t, \n, \r that will be spaces.
            false
        } else {
            // https://unicode.org/reports/tr44/#GC_Values_Table
            // Other Cc | Cf | Cs | Co | Cn
            isOther(c.properties.generalCategory)
        }
    }

    private func isOther(_ c: Unicode.GeneralCategory) -> Bool {
        c == .control || c == .format || c == .surrogate || c == .privateUse || c == .unassigned
    }

    private func handleChineseChars(text: String) -> String {
        text.map { c in
            if let scalar = c.unicodeScalars.first, Utils.isChineseChar(scalar) {
                " \(c) "
            } else {
                "\(c)"
            }
        }
        .joined()
    }

    private func stripAccents(text: String) -> String {
        // This might be the same as `text.folding(options: .diacriticInsensitive, locale: nil)`
        String(text.decomposedStringWithCanonicalMapping.unicodeScalars.filter { scalar in
            !(scalar.value >= 0x0300 && scalar.value <= 0x036F)
        })
    }
}

class PrecompiledNormalizer: Normalizer {
    // TODO: use `precompiledCharsmap` (base64-encoded string) from the configuration
    required init(config: Config) { }

    func normalize(text: String) -> String {
        // TODO: This is a simplified implementation.
        // - The following comments also apply here:
        // https://github.com/xenova/transformers.js/blob/main/src/tokenizers.js#L2237-L2247
        // - For a proper implementation, see:
        // https://github.com/huggingface/tokenizers/blob/b58227c7f1ccf8b73ee2268354336da56d91e492/tokenizers/src/normalizers/precompiled.rs#L36
        var output = ""
        var hasFullwidthTilde = false

        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x0001...0x0008, 0x000B, 0x000E...0x001F, 0x007F, 0x008F, 0x009F:
                // Non-printing control characters
                output.append("")
            case 0x0009, 0x000A, 0x000C, 0x000D, 0x1680, 0x200B...0x200F, 0x2028, 0x2029, 0x2581,
                 0xFEFF, 0xFFFD:
                // Separators
                output.append(" ")
            case 0xFF5E:
                hasFullwidthTilde = true
                fallthrough
            default:
                output.append(Character(scalar))
            }
        }

        if hasFullwidthTilde {
            return
                output
                    .split(by: "\u{FF5E}")
                    .map { $0.precomposedStringWithCompatibilityMapping }
                    .joined(separator: "\u{FF5E}")
        } else {
            return output.precomposedStringWithCompatibilityMapping
        }
    }
}

class StripAccentsNormalizer: Normalizer {
    required init(config: Config) { }

    func normalize(text: String) -> String {
        text.precomposedStringWithCompatibilityMapping
    }
}

class StripNormalizer: Normalizer {
    let leftStrip: Bool
    let rightStrip: Bool

    required init(config: Config) {
        leftStrip = config.stripLeft.boolean(or: true)
        rightStrip = config.stripRight.boolean(or: true)
    }

    func normalize(text: String) -> String {
        var result = text

        if leftStrip {
            result = String(result.drop(while: { $0.isWhitespace }))
        }

        if rightStrip {
            result = String(result.reversed().drop(while: { $0.isWhitespace }).reversed())
        }

        return result
    }
}

enum StringReplacePattern {
    case regexp(regexp: NSRegularExpression, replacement: String)
    case string(pattern: String, replacement: String)
}

extension StringReplacePattern {
    func replace(_ text: String) -> String {
        switch self {
        case let .regexp(regexp, replacement):
            let range = NSRange(text.startIndex..., in: text)
            let replaced = regexp.stringByReplacingMatches(
                in: text, options: [], range: range, withTemplate: replacement
            )
            return replaced
        case let .string(toReplace, replacement):
            return text.replacingOccurrences(of: toReplace, with: replacement)
        }
    }
}

extension StringReplacePattern {
    static func from(config: Config) -> StringReplacePattern? {
        guard let replacement = config.content.string() else { return nil }
        if let pattern = config.pattern.String.string() {
            return StringReplacePattern.string(pattern: pattern, replacement: replacement)
        }
        if let pattern = config.pattern.Regex.string() {
            guard let regexp = try? NSRegularExpression(pattern: pattern, options: []) else {
                fatalError("Cannot build regexp from \(pattern)")
            }
            return StringReplacePattern.regexp(regexp: regexp, replacement: replacement)
        }
        return nil
    }
}

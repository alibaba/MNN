//
//  ReplacementContentType.swift
//  MarkdownView
//
//  Created by 秋星桥 on 7/5/25.
//

import Foundation

public extension MarkdownParser {
    static func replacementText(for contentType: String, identifier: String) -> String {
        "`" + "md://content?type=\(contentType)&identifier=\(identifier)" + "`"
    }

    enum ReplacementContentType: String {
        case unknown
        case math
    }

    static func replacementText(for contentType: ReplacementContentType, identifier: String) -> String {
        replacementText(for: contentType.rawValue, identifier: identifier)
    }

    static func typeForReplacementText(_ text: String, allowEnclosingCharacterMismatch: Bool = true) -> ReplacementContentType {
        if !allowEnclosingCharacterMismatch {
            guard text.hasPrefix("`"), text.hasSuffix("`") else { return .unknown }
        }
        let text = text.trimmingCharacters(in: .init(charactersIn: "`"))
        guard let comps = URLComponents(string: text) else {
            return .unknown
        }
        for queryItem in comps.queryItems ?? [] where queryItem.name == "type" {
            guard let value = queryItem.value else { return .unknown }
            guard let type = ReplacementContentType(rawValue: value) else { return .unknown }
            return type
        }
        return .unknown
    }

    static func identifierForReplacementText(_ text: String, allowEnclosingCharacterMismatch: Bool = true) -> String? {
        if !allowEnclosingCharacterMismatch {
            guard text.hasPrefix("`"), text.hasSuffix("`") else { return nil }
        }
        let text = text.trimmingCharacters(in: .init(charactersIn: "`"))
        assert(text.hasPrefix("md://content?type="))
        guard let comps = URLComponents(string: text) else {
            assertionFailure()
            return nil
        }
        for queryItem in comps.queryItems ?? [] where queryItem.name == "identifier" {
            guard let value = queryItem.value else {
                assertionFailure()
                return nil
            }
            return value
        }
        assertionFailure()
        return nil
    }
}

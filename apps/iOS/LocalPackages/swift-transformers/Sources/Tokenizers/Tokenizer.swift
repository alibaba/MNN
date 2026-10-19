//
//  Tokenizer.swift
//
//
//  Created by Pedro Cuenca on 6/5/23.
//

import Foundation
import Hub
import Jinja

public typealias Message = [String: Any]
public typealias ToolSpec = [String: Any]

public enum TokenizerError: LocalizedError {
    case missingConfig
    case missingTokenizerClassInConfig
    case unsupportedTokenizer(String)
    case missingVocab
    case malformedVocab
    case chatTemplate(String)
    case missingChatTemplate
    case tooLong(String)
    case mismatchedConfig(String)

    public var errorDescription: String? {
        switch self {
        case .missingConfig:
            String(localized: "Tokenizer configuration is missing.", comment: "Error when tokenizer config cannot be found")
        case .missingTokenizerClassInConfig:
            String(localized: "The tokenizer class is not specified in the configuration.", comment: "Error when tokenizer_class is missing in config")
        case let .unsupportedTokenizer(name):
            String(localized: "The tokenizer type '\(name)' is not supported.", comment: "Error when tokenizer type is not supported")
        case .missingVocab:
            String(localized: "Vocabulary file is missing from the tokenizer configuration.", comment: "Error when vocab file is missing")
        case .malformedVocab:
            String(localized: "The vocabulary file is malformed or corrupted.", comment: "Error when vocab file is malformed")
        case let .chatTemplate(message):
            String(localized: "Chat template error: \(message)", comment: "Error with chat template")
        case .missingChatTemplate:
            String(localized: "This tokenizer does not have a chat template, and no template was passed.")
        case let .tooLong(message):
            String(localized: "Input is too long: \(message)", comment: "Error when input exceeds maximum length")
        case let .mismatchedConfig(message):
            String(localized: "Tokenizer configuration mismatch: \(message)", comment: "Error when tokenizer configuration is inconsistent")
        }
    }
}

public protocol TokenizingModel {
    func tokenize(text: String) -> [String]

    /// Alias for `tokenize`
    func callAsFunction(_ text: String) -> [String]

    func convertTokenToId(_ token: String) -> Int?
    func convertTokensToIds(_ tokens: [String]) -> [Int?]

    func convertIdToToken(_ id: Int) -> String?
    func convertIdsToTokens(_ ids: [Int]) -> [String?]

    var bosToken: String? { get }
    var bosTokenId: Int? { get }
    var eosToken: String? { get }
    var eosTokenId: Int? { get }
    var unknownToken: String? { get }
    var unknownTokenId: Int? { get }

    var fuseUnknownTokens: Bool { get }
}

/// Helper - possibly to be moved somewhere else
func addedTokenAsString(_ addedToken: Config?) -> String? {
    guard let addedToken else { return nil }
    if let stringValue = addedToken.string() {
        return stringValue
    }
    // This is possibly a serialization of the AddedToken class
    // TODO: support lstrip, rstrip, normalized, etc.
    return addedToken.content.string()
}

public extension TokenizingModel {
    func callAsFunction(_ text: String) -> [String] {
        tokenize(text: text)
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        ids.map { convertIdToToken($0) }
    }
}

/// A tokenizer model that is set up with Hub configuration data
public protocol PreTrainedTokenizerModel: TokenizingModel {
    init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws
}

struct TokenizerModel {
    static let knownTokenizers: [String: PreTrainedTokenizerModel.Type] = [
        "BertTokenizer": BertTokenizer.self,
        "DistilbertTokenizer": BertTokenizer.self,
        "DistilBertTokenizer": BertTokenizer.self,
        "RobertaTokenizer": BPETokenizer.self,
        "CodeGenTokenizer": CodeGenTokenizer.self,
        "CodeLlamaTokenizer": CodeLlamaTokenizer.self,
        "FalconTokenizer": FalconTokenizer.self,
        "GemmaTokenizer": GemmaTokenizer.self,
        "GPT2Tokenizer": GPT2Tokenizer.self,
        "LlamaTokenizer": LlamaTokenizer.self,
        "T5Tokenizer": T5Tokenizer.self,
        "WhisperTokenizer": WhisperTokenizer.self,
        "CohereTokenizer": CohereTokenizer.self,
        "Qwen2Tokenizer": Qwen2Tokenizer.self,
        "PreTrainedTokenizer": BPETokenizer.self,
    ]

    static func unknownToken(from tokenizerConfig: Config) -> String? {
        tokenizerConfig.unkToken.content.string() ?? tokenizerConfig.unkToken.string()
    }

    public static func from(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws -> TokenizingModel {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass.string() else {
            throw TokenizerError.missingTokenizerClassInConfig
        }

        // Some tokenizer_class entries use a Fast suffix
        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
        guard let tokenizerClass = TokenizerModel.knownTokenizers[tokenizerName] else {
            throw TokenizerError.unsupportedTokenizer(tokenizerName)
        }

        return try tokenizerClass.init(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens)
    }
}

public enum ChatTemplateArgument {
    /// A Jinja template to use for the conversation. Normally it is not necessary to provide a template, since it will be read from the tokenizer config.
    case literal(String)
    /// For models whose tokenizer config includes multiple chat templates, the template can be specified by name. Normally this is not necessary.
    case name(String)
}

public protocol Tokenizer {
    func tokenize(text: String) -> [String]

    /// Main entry point
    func encode(text: String) -> [Int]
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int]

    /// Decode
    func decode(tokens: [Int]) -> String
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String

    func convertTokenToId(_ token: String) -> Int?
    func convertTokensToIds(_ tokens: [String]) -> [Int?]

    func convertIdToToken(_ id: Int) -> String?
    func convertIdsToTokens(_ ids: [Int]) -> [String?]

    var bosToken: String? { get }
    var bosTokenId: Int? { get }
    var eosToken: String? { get }
    var eosTokenId: Int? { get }
    var unknownToken: String? { get }
    var unknownTokenId: Int? { get }

    var hasChatTemplate: Bool { get }

    /// The appropriate chat template is selected from the tokenizer config
    func applyChatTemplate(messages: [Message]) throws -> [Int]

    /// The appropriate chat template is selected from the tokenizer config
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int]

    /// The appropriate chat template is selected from the tokenizer config
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?, additionalContext: [String: Any]?) throws -> [Int]

    /// The chat template is provided as a string literal or specified by name
    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int]

    /// The chat template is provided as a string literal
    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int]

    func applyChatTemplate(
        messages: [Message],
        // A chat template can optionally be provided or specified by name when several templates are included in the tokenizer config. Normally this is not necessary.
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?
    ) throws -> [Int]

    func applyChatTemplate(
        messages: [Message],
        // A chat template can optionally be provided or specified by name when several templates are included in the tokenizer config. Normally this is not necessary.
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: Any]?
    ) throws -> [Int]
}

extension Tokenizer {
    public var hasChatTemplate: Bool { false }

    /// Call previous signature for backwards compatibility
    func applyChatTemplate(
        messages: [Message],
        // A chat template can optionally be provided or specified by name when several templates are included in the tokenizer config. Normally this is not necessary.
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: Any]?
    ) throws -> [Int] {
        if additionalContext == nil {
            try applyChatTemplate(
                messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: addGenerationPrompt, truncation: truncation, maxLength: maxLength,
                tools: tools
            )
        } else {
            throw TokenizerError.chatTemplate("Not implemented")
        }
    }
}

public extension Tokenizer {
    func callAsFunction(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokens: [Int]) -> String {
        decode(tokens: tokens, skipSpecialTokens: false)
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        ids.map { convertIdToToken($0) }
    }
}

let specialTokenAttributes: [String] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
]

public class PreTrainedTokenizer: Tokenizer {
    let model: TokenizingModel

    public var bosToken: String? { model.bosToken }
    public var bosTokenId: Int? { model.bosTokenId }
    public var eosToken: String? { model.eosToken }
    public var eosTokenId: Int? { model.eosTokenId }
    public var unknownToken: String? { model.unknownToken }
    public var unknownTokenId: Int? { model.unknownTokenId }
    public var fuseUnknownTokens: Bool { model.fuseUnknownTokens }

    private let addedTokens: Set<String>
    private let specialTokens: [String: Int]
    private let addedTokensRegex: NSRegularExpression?

    private let preTokenizer: PreTokenizer?
    private let normalizer: Normalizer?
    private let postProcessor: PostProcessor?
    private let decoder: Decoder?
    private let tokenizerConfig: Config

    private let cleanUpTokenizationSpaces: Bool

    public required init(tokenizerConfig: Config, tokenizerData: Config) throws {
        var addedTokens: [String: Int] = [:]
        var specialTokens: [String: Int] = [:]
        for addedToken in tokenizerData["addedTokens"].array(or: []) {
            guard let id = addedToken["id"].integer() else { continue /* malformed: token with no id */ }
            guard let content = addedToken.content.string() else { continue /* malformed: token with no content */ }
            addedTokens[content] = id

            if addedToken["special"].boolean(or: false) {
                specialTokens[content] = id
            }
        }

        // Convert to tuples for easier access, then sort by length (descending) to avoid early partial matches
        // (https://github.com/xenova/transformers.js/commit/c305c3824f628f1f02806a6310bd3b18b0f7f8f5)
        let unwrappedAddedTokens: [(content: String, prefix: Bool, suffix: Bool)] = (tokenizerData["addedTokens"].array(or: [])).compactMap { addedToken -> (String, Bool, Bool)? in
            guard let content = addedToken.content.string() else { return nil }
            let prefix = addedToken["lstrip"].boolean(or: false)
            let suffix = addedToken["rstrip"].boolean(or: false)
            return (content: content, prefix: prefix, suffix: suffix)
        }.sorted {
            $0.content.count > $1.content.count
        }

        // then concatenate into regular expression
        let addedTokensRegexString = unwrappedAddedTokens.map {
            let token = NSRegularExpression.escapedPattern(for: $0.content)
            let prefix = $0.prefix ? #"\s*"# : ""
            let suffix = $0.suffix ? #"\s*"# : ""
            return "\(prefix)(\(token))\(suffix)"
        }.joined(separator: "|")
        addedTokensRegex = try? NSRegularExpression(pattern: addedTokensRegexString, options: [])

        // TODO: specialTokens are stored but never used
        self.specialTokens = specialTokens
        self.addedTokens = Set(addedTokens.keys)

        preTokenizer = PreTokenizerFactory.fromConfig(config: tokenizerData["preTokenizer"])
        normalizer = NormalizerFactory.fromConfig(config: tokenizerData["normalizer"])
        postProcessor = PostProcessorFactory.fromConfig(config: tokenizerData["postProcessor"])
        decoder = DecoderFactory.fromConfig(config: tokenizerData["decoder"], addedTokens: self.addedTokens)
        cleanUpTokenizationSpaces = tokenizerConfig.cleanUpTokenizationSpaces.boolean(or: true)
        self.tokenizerConfig = tokenizerConfig

        model = try TokenizerModel.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens)
    }

    func preTokenize(_ text: String, options: PreTokenizerOptions) -> [String] {
        guard let preTokenizer else { return [text] }
        return preTokenizer(text: text, options: options)
    }

    func normalize(_ text: String) -> String {
        guard let normalizer else { return text }
        return normalizer(text: text)
    }

    func postProcess(_ tokens: [String], addSpecialTokens: Bool = true) -> [String] {
        guard let postProcessor else { return tokens }
        return postProcessor(tokens: tokens, addSpecialTokens: addSpecialTokens)
    }

    func decodeTokens(_ tokens: [String]) -> [String] {
        guard let tokenDecoder = decoder else { return tokens }
        return tokenDecoder(tokens: tokens)
    }

    /// Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms
    func cleanUp(text: String) -> String {
        guard cleanUpTokenizationSpaces else { return text }

        return
            text
                .replacingOccurrences(of: " .", with: ".")
                .replacingOccurrences(of: " ?", with: "?")
                .replacingOccurrences(of: " !", with: "!")
                .replacingOccurrences(of: " ,", with: ",")
                .replacingOccurrences(of: " ' ", with: "'")
                .replacingOccurrences(of: " n't", with: "n't")
                .replacingOccurrences(of: " 'm", with: "'m")
                .replacingOccurrences(of: " 's", with: "'s")
                .replacingOccurrences(of: " 've", with: "'ve")
                .replacingOccurrences(of: " 're", with: "'re")
    }

    func fuseUnknown(_ tokens: [String]) -> [String] {
        guard fuseUnknownTokens else { return tokens }
        let (fused, _) = tokens.reduce((fused: [String](), previousIsUnknown: false)) { result, token in
            var (fused, previousIsUnknown) = result
            let isUnknown = model.convertTokenToId(token) == model.unknownTokenId
            if isUnknown {
                if !previousIsUnknown { fused.append(token) }
            } else {
                fused.append(token)
            }
            return (fused, isUnknown)
        }
        return fused
    }

    public func tokenize(text: String) -> [String] {
        // Take care of special tokens first
        let sections: [String] = if let regex = addedTokensRegex {
            text.split(by: regex)
        } else {
            [text]
        }
        return sections.enumerated().map { section, x in
            if addedTokens.contains(x) { return [x] }
            return preTokenize(normalize(x), options: section == 0 ? [.firstSection] : []).flatMap { model($0) }
        }.flatMap { fuseUnknown($0) }
    }

    /// Main entry point
    public func encode(text: String, addSpecialTokens: Bool = true) -> [Int] {
        postProcess(tokenize(text: text), addSpecialTokens: addSpecialTokens).map { model.convertTokenToId($0)! }
    }

    public func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    public func decode(tokens: [Int], skipSpecialTokens: Bool = false) -> String {
        // IDs to tokens
        let tokenStrings: [String]
        if skipSpecialTokens {
            let specialTokenIDs = Set(specialTokens.values)
            tokenStrings =
                tokens
                    .filter { !specialTokenIDs.contains($0) }
                    .compactMap { model.convertIdToToken($0) }
        } else {
            tokenStrings = tokens.compactMap { model.convertIdToToken($0) }
        }
        let decoded = decodeTokens(tokenStrings)
        // At this point we should have a single String
        return cleanUp(text: decoded.joined(separator: ""))
    }

    public func convertTokenToId(_ token: String) -> Int? {
        model.convertTokenToId(token)
    }

    public func convertIdToToken(_ id: Int) -> String? {
        model.convertIdToToken(id)
    }

    public var hasChatTemplate: Bool {
        !tokenizerConfig.chatTemplate.isNull()
    }

    public func applyChatTemplate(messages: [Message]) throws -> [Int] {
        try applyChatTemplate(messages: messages, addGenerationPrompt: true)
    }

    public func applyChatTemplate(messages: [Message], tools: [ToolSpec]? = nil) throws -> [Int] {
        try applyChatTemplate(messages: messages, addGenerationPrompt: true, tools: tools)
    }

    public func applyChatTemplate(messages: [Message], tools: [ToolSpec]? = nil, additionalContext: [String: Any]? = nil) throws
        -> [Int]
    {
        try applyChatTemplate(
            messages: messages,
            addGenerationPrompt: true,
            tools: tools,
            additionalContext: additionalContext
        )
    }

    public func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: true)
    }

    public func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: .literal(chatTemplate), addGenerationPrompt: true)
    }

    public func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument? = nil,
        addGenerationPrompt: Bool = false,
        truncation: Bool = false,
        maxLength: Int? = nil,
        tools: [ToolSpec]? = nil
    ) throws -> [Int] {
        try applyChatTemplate(
            messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: addGenerationPrompt, truncation: truncation, maxLength: maxLength,
            tools: tools, additionalContext: nil
        )
    }

    public func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument? = nil,
        addGenerationPrompt: Bool = false,
        truncation: Bool = false,
        maxLength: Int? = nil,
        // A list of tools (callable functions) that will be accessible to the model. If the template does not
        // support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
        // giving the name, description and argument types for the tool. See the
        // [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
        // for more information.
        tools: [ToolSpec]? = nil,
        additionalContext: [String: Any]? = nil
    ) throws -> [Int] {
        var selectedChatTemplate: String?
        if let chatTemplate, case let .literal(template) = chatTemplate {
            // Use chat template from argument
            selectedChatTemplate = template
        } else if !tokenizerConfig.chatTemplate.isNull() {
            let valueFromConfig: Config = tokenizerConfig.chatTemplate
            if let arrayValue = valueFromConfig.array() {
                // If the config specifies a list of chat templates, convert them to a dictionary
                let templateDict = [String: String](
                    uniqueKeysWithValues: arrayValue.compactMap { item in
                        guard let name = item["name"].string(), let template = item["template"].string() else {
                            return nil
                        }
                        return (name, template)
                    })
                if let chatTemplate, case let .name(name) = chatTemplate {
                    // Select chat template from config by name
                    if let matchingDictEntry = templateDict[name] {
                        selectedChatTemplate = matchingDictEntry
                    } else {
                        throw TokenizerError.chatTemplate("No chat template named \"\(name)\" was found in the tokenizer config")
                    }
                } else if let tools, !tools.isEmpty, let toolUseTemplate = templateDict["tool_use"] {
                    // Use tool use chat template from config
                    selectedChatTemplate = toolUseTemplate
                } else if let defaultChatTemplate = templateDict["default"] {
                    // Use default chat template from config
                    selectedChatTemplate = defaultChatTemplate
                }
            } else if let stringValue = valueFromConfig.string() {
                // Use chat template from config
                selectedChatTemplate = stringValue
            }
        }

        guard let selectedChatTemplate else {
            throw TokenizerError.missingChatTemplate
        }

        let template = try Template(selectedChatTemplate)
        var context: [String: Any] = [
            "messages": messages,
            "add_generation_prompt": addGenerationPrompt,
        ]
        if let tools {
            context["tools"] = tools
        }
        if let additionalContext {
            /*
             Additional keys and values to be added to the context provided to the prompt templating engine.
             For example, the app could set "tools_in_user_message" to false for Llama 3.1 and 3.2 if a system message is provided.
             The default value is true in the Llama 3.1 and 3.2 chat templates, but these models will perform better if the tools are included in a system message.
             */
            for (key, value) in additionalContext {
                context[key] = value
            }
        }

        for (key, value) in tokenizerConfig.dictionary(or: [:]) {
            if specialTokenAttributes.contains(key.string), !value.isNull() {
                if let stringValue = value.string() {
                    context[key.string] = stringValue
                } else if let dictionary = value.dictionary() {
                    context[key.string] = addedTokenAsString(Config(dictionary))
                } else if let array: [String] = value.get() {
                    context[key.string] = array
                } else {
                    context[key.string] = value
                }
            }
        }

        let rendered = try template.render(context)
        var encodedTokens = encode(text: rendered, addSpecialTokens: false)
        var maxLength = maxLength ?? encodedTokens.count
        maxLength = min(maxLength, tokenizerConfig.modelMaxLength.integer() ?? maxLength)
        if encodedTokens.count > maxLength {
            if truncation {
                encodedTokens = Array(encodedTokens.prefix(maxLength))
            }
        }

        return encodedTokens
    }
}

// MARK: - Building

public struct AutoTokenizer { }

struct PreTrainedTokenizerClasses {
    /// Class overrides for custom behaviour
    /// Not to be confused with the TokenizerModel classes defined in TokenizerModel
    static let tokenizerClasses: [String: PreTrainedTokenizer.Type] = [
        "LlamaTokenizer": LlamaPreTrainedTokenizer.self,
    ]
}

public extension AutoTokenizer {
    internal static func tokenizerClass(for tokenizerConfig: Config) -> PreTrainedTokenizer.Type {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass.string() else {
            return PreTrainedTokenizer.self
        }

        // Some tokenizer_class entries use a Fast suffix
        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
        if let tokenizerClass = PreTrainedTokenizerClasses.tokenizerClasses[tokenizerName] {
            return tokenizerClass
        }

        return PreTrainedTokenizer.self
    }

    static func from(tokenizerConfig: Config, tokenizerData: Config) throws -> Tokenizer {
        let tokenizerClass = tokenizerClass(for: tokenizerConfig)
        return try tokenizerClass.init(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }

    static func from(
        pretrained model: String,
        hubApi: HubApi = .shared
    ) async throws -> Tokenizer {
        let config = LanguageModelConfigurationFromHub(modelName: model, hubApi: hubApi)
        guard let tokenizerConfig = try await config.tokenizerConfig else { throw TokenizerError.missingConfig }
        let tokenizerData = try await config.tokenizerData

        return try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }

    static func from(
        modelFolder: URL,
        hubApi: HubApi = .shared
    ) async throws -> Tokenizer {
        let config = LanguageModelConfigurationFromHub(modelFolder: modelFolder, hubApi: hubApi)
        guard let tokenizerConfig = try await config.tokenizerConfig else { throw TokenizerError.missingConfig }
        let tokenizerData = try await config.tokenizerData

        return try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
    }
}

// MARK: - Tokenizer model classes

class GPT2Tokenizer: BPETokenizer { }
class FalconTokenizer: BPETokenizer { }
class LlamaTokenizer: BPETokenizer { }
class CodeGenTokenizer: BPETokenizer { }
class WhisperTokenizer: BPETokenizer { }
class GemmaTokenizer: BPETokenizer { }
class CodeLlamaTokenizer: BPETokenizer { }
class CohereTokenizer: BPETokenizer { }
class Qwen2Tokenizer: BPETokenizer { }

class T5Tokenizer: UnigramTokenizer { }

// MARK: - PreTrainedTokenizer classes

let sentencePieceUnderline = "▁"

/// Hack for Llama tokenizers, see https://github.com/huggingface/transformers/blob/bcb841f0073fcd7a4fb88ea8064313c17dcab04a/src/transformers/models/llama/tokenization_llama_fast.py#L181
/// Return updated config, or nil
func maybeUpdatePostProcessor(tokenizerConfig: Config, processorConfig: Config?) throws -> Config? {
    // If it's already a Template processor (instead of a ByteLevel one), assume it's correct
    let postProcessor = PostProcessorFactory.fromConfig(config: processorConfig)
    guard !(postProcessor is TemplateProcessing) else { return nil }

    let addBosToken = tokenizerConfig.addBosToken.boolean(or: false)
    let bosToken = addedTokenAsString(tokenizerConfig.bosToken)
    if addBosToken, bosToken == nil {
        throw TokenizerError.mismatchedConfig("add_bos_token is True but bos_token is nil")
    }

    let addEosToken = tokenizerConfig.addEosToken.boolean(or: false)
    let eosToken = addedTokenAsString(tokenizerConfig.eosToken)
    if addEosToken, eosToken == nil {
        throw TokenizerError.mismatchedConfig("add_eos_token is True but eos_token is nil")
    }

    // alt implementation
    var single: [[String: Any]] = []
    if addBosToken {
        single = single + [["SpecialToken": ["id": bosToken!, "type_id": 0]]]
    }
    single = single + [["Sequence": ["id": "A", "type_id": 0]]]
    if addEosToken {
        single = single + [["SpecialToken": ["id": eosToken!, "type_id": 0]]]
    }

    var pair: [[String: Any]] = single
    if addBosToken {
        pair = pair + [["SpecialToken": ["id": bosToken!, "type_id": 1]]]
    }
    pair = pair + [["Sequence": ["id": "B", "type_id": 1]]]
    if addEosToken {
        pair = pair + [["SpecialToken": ["id": eosToken!, "type_id": 1]]]
    }

    let postProcessorConfig = Config(["type": PostProcessorType.TemplateProcessing.rawValue, "single": single, "pair": pair])
    return postProcessorConfig
}

/// See https://github.com/xenova/transformers.js/blob/1a9964fb09b8f54fcbeac46dc6aae8d76795809d/src/tokenizers.js#L3203 for these exceptions
class LlamaPreTrainedTokenizer: PreTrainedTokenizer {
    let isLegacy: Bool

    required init(tokenizerConfig: Config, tokenizerData: Config) throws {
        isLegacy = tokenizerConfig.legacy.boolean(or: true)
        var configDictionary = tokenizerData.dictionary(or: [:])
        if !isLegacy {
            _ = configDictionary.removeValue(forKey: "normalizer")
            configDictionary["pre_tokenizer"] = [
                "type": "Metaspace", "replacement": .init(sentencePieceUnderline), "add_prefix_space": true, "prepend_scheme": "first",
            ]
        }

        if let postProcessorConfig = try maybeUpdatePostProcessor(tokenizerConfig: tokenizerConfig, processorConfig: tokenizerData["postProcessor"]) {
            configDictionary["post_processor"] = .init(postProcessorConfig.dictionary(or: [:]))
        }

        let updatedData = Config(configDictionary)
        try super.init(tokenizerConfig: tokenizerConfig, tokenizerData: updatedData)
    }
}

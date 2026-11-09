//
//  Hub.swift
//
//
//  Created by Pedro Cuenca on 18/5/23.
//

import Foundation

public struct Hub { }

public extension Hub {
    enum HubClientError: LocalizedError {
        case authorizationRequired
        case httpStatusCode(Int)
        case parse
        case unexpectedError
        case downloadError(String)
        case fileNotFound(String)
        case networkError(URLError)
        case resourceNotFound(String)
        case configurationMissing(String)
        case fileSystemError(Error)
        case parseError(String)

        public var errorDescription: String? {
            switch self {
            case .authorizationRequired:
                String(localized: "Authentication required. Please provide a valid Hugging Face token.")
            case let .httpStatusCode(code):
                String(localized: "HTTP error with status code: \(code)")
            case .parse:
                String(localized: "Failed to parse server response.")
            case .unexpectedError:
                String(localized: "An unexpected error occurred.")
            case let .downloadError(message):
                String(localized: "Download failed: \(message)")
            case let .fileNotFound(filename):
                String(localized: "File not found: \(filename)")
            case let .networkError(error):
                String(localized: "Network error: \(error.localizedDescription)")
            case let .resourceNotFound(resource):
                String(localized: "Resource not found: \(resource)")
            case let .configurationMissing(file):
                String(localized: "Required configuration file missing: \(file)")
            case let .fileSystemError(error):
                String(localized: "File system error: \(error.localizedDescription)")
            case let .parseError(message):
                String(localized: "Parse error: \(message)")
            }
        }
    }

    enum RepoType: String, Codable {
        case models
        case datasets
        case spaces
    }

    struct Repo: Codable {
        public let id: String
        public let type: RepoType

        public init(id: String, type: RepoType = .models) {
            self.id = id
            self.type = type
        }
    }
}

public class LanguageModelConfigurationFromHub {
    struct Configurations {
        var modelConfig: Config
        var tokenizerConfig: Config?
        var tokenizerData: Config
    }

    private var configPromise: Task<Configurations, Error>?

    public init(
        modelName: String,
        revision: String = "main",
        hubApi: HubApi = .shared
    ) {
        configPromise = Task.init {
            try await self.loadConfig(modelName: modelName, revision: revision, hubApi: hubApi)
        }
    }

    public init(
        modelFolder: URL,
        hubApi: HubApi = .shared
    ) {
        configPromise = Task {
            try await self.loadConfig(modelFolder: modelFolder, hubApi: hubApi)
        }
    }

    public var modelConfig: Config {
        get async throws {
            try await configPromise!.value.modelConfig
        }
    }

    public var tokenizerConfig: Config? {
        get async throws {
            if let hubConfig = try await configPromise!.value.tokenizerConfig {
                // Try to guess the class if it's not present and the modelType is
                if let _: String = hubConfig.tokenizerClass?.string() { return hubConfig }
                guard let modelType = try await modelType else { return hubConfig }

                // If the config exists but doesn't contain a tokenizerClass, use a fallback config if we have it
                if let fallbackConfig = Self.fallbackTokenizerConfig(for: modelType) {
                    let configuration = fallbackConfig.dictionary()?.merging(hubConfig.dictionary(or: [:]), strategy: { current, _ in current }) ?? [:]
                    return Config(configuration)
                }

                // Guess by capitalizing
                var configuration = hubConfig.dictionary(or: [:])
                configuration["tokenizer_class"] = .init("\(modelType.capitalized)Tokenizer")
                return Config(configuration)
            }

            // Fallback tokenizer config, if available
            guard let modelType = try await modelType else { return nil }
            return Self.fallbackTokenizerConfig(for: modelType)
        }
    }

    public var tokenizerData: Config {
        get async throws {
            try await configPromise!.value.tokenizerData
        }
    }

    public var modelType: String? {
        get async throws {
            try await modelConfig.modelType.string()
        }
    }

    func loadConfig(
        modelName: String,
        revision: String,
        hubApi: HubApi = .shared
    ) async throws -> Configurations {
        let filesToDownload = ["config.json", "tokenizer_config.json", "chat_template.jinja", "chat_template.json", "tokenizer.json"]
        let repo = Hub.Repo(id: modelName)

        do {
            let downloadedModelFolder = try await hubApi.snapshot(from: repo, revision: revision, matching: filesToDownload)
            return try await loadConfig(modelFolder: downloadedModelFolder, hubApi: hubApi)
        } catch {
            // Convert generic errors to more specific ones
            if let urlError = error as? URLError {
                switch urlError.code {
                case .notConnectedToInternet, .networkConnectionLost:
                    throw Hub.HubClientError.networkError(urlError)
                case .resourceUnavailable:
                    throw Hub.HubClientError.resourceNotFound(modelName)
                default:
                    throw Hub.HubClientError.networkError(urlError)
                }
            } else {
                throw error
            }
        }
    }

    func loadConfig(
        modelFolder: URL,
        hubApi: HubApi = .shared
    ) async throws -> Configurations {
        do {
            // Load required configurations
            let modelConfigURL = modelFolder.appending(path: "config.json")
            guard FileManager.default.fileExists(atPath: modelConfigURL.path) else {
                throw Hub.HubClientError.configurationMissing("config.json")
            }

            let modelConfig = try hubApi.configuration(fileURL: modelConfigURL)

            let tokenizerDataURL = modelFolder.appending(path: "tokenizer.json")
            guard FileManager.default.fileExists(atPath: tokenizerDataURL.path) else {
                throw Hub.HubClientError.configurationMissing("tokenizer.json")
            }

            let tokenizerData = try hubApi.configuration(fileURL: tokenizerDataURL)

            // Load tokenizer config (optional)
            var tokenizerConfig: Config? = nil
            let tokenizerConfigURL = modelFolder.appending(path: "tokenizer_config.json")
            if FileManager.default.fileExists(atPath: tokenizerConfigURL.path) {
                tokenizerConfig = try hubApi.configuration(fileURL: tokenizerConfigURL)
            }

            // Check for chat template and merge if available
            // Prefer .jinja template over .json template
            var chatTemplate: String? = nil
            let chatTemplateJinjaURL = modelFolder.appending(path: "chat_template.jinja")
            let chatTemplateJsonURL = modelFolder.appending(path: "chat_template.json")

            if FileManager.default.fileExists(atPath: chatTemplateJinjaURL.path) {
                // Try to load .jinja template as plain text
                chatTemplate = try? String(contentsOf: chatTemplateJinjaURL, encoding: .utf8)
            } else if FileManager.default.fileExists(atPath: chatTemplateJsonURL.path),
                      let chatTemplateConfig = try? hubApi.configuration(fileURL: chatTemplateJsonURL)
            {
                // Fall back to .json template
                chatTemplate = chatTemplateConfig.chatTemplate.string()
            }

            if let chatTemplate {
                // Create or update tokenizer config with chat template
                if var configDict = tokenizerConfig?.dictionary() {
                    configDict["chat_template"] = .init(chatTemplate)
                    tokenizerConfig = Config(configDict)
                } else {
                    tokenizerConfig = Config(["chat_template": chatTemplate])
                }
            }

            return Configurations(
                modelConfig: modelConfig,
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData
            )
        } catch let error as Hub.HubClientError {
            throw error
        } catch {
            if let nsError = error as NSError? {
                if nsError.domain == NSCocoaErrorDomain, nsError.code == NSFileReadNoSuchFileError {
                    throw Hub.HubClientError.fileSystemError(error)
                } else if nsError.domain == "NSJSONSerialization" {
                    throw Hub.HubClientError.parseError("Invalid JSON format: \(nsError.localizedDescription)")
                }
            }
            throw Hub.HubClientError.fileSystemError(error)
        }
    }

    static func fallbackTokenizerConfig(for modelType: String) -> Config? {
        guard let url = Bundle.module.url(forResource: "\(modelType)_tokenizer_config", withExtension: "json") else {
            return nil
        }

        do {
            let data = try Data(contentsOf: url)
            let parsed = try JSONSerialization.jsonObject(with: data, options: [])
            guard let dictionary = parsed as? [NSString: Any] else {
                throw Hub.HubClientError.parseError("Failed to parse fallback tokenizer config")
            }
            return Config(dictionary)
        } catch let error as Hub.HubClientError {
            print("Error loading fallback tokenizer config: \(error.localizedDescription)")
            return nil
        } catch {
            print("Error loading fallback tokenizer config: \(error.localizedDescription)")
            return nil
        }
    }
}

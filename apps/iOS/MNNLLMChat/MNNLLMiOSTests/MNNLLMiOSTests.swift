//
//  MNNLLMiOSTests.swift
//  MNNLLMiOSTests
//
//  Created by 游薪渝(揽清) on 2024/12/26.
//

@testable import ExyteChat
@testable import ExyteMediaPicker
import ImageIO
import CMNNInference
import Combine
import Foundation
import SwiftUI
import Testing
import UIKit
@testable import MNNChatCore

@Suite(.serialized)
@MainActor
struct MNNLLMiOSTests {
    @Test func highResolutionImageInputStaysWithinMobileMemory() async throws {
        try await waitUntil(attempts: 500) { UIApplication.shared.applicationState == .active }
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("HighResolutionImageQA-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let catURL = try #require(Bundle.main.resourceURL).appendingPathComponent("LocalModel/preset_images/cat.jpg")
        let cat = try #require(UIImage(contentsOfFile: catURL.path))
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let photo = UIGraphicsImageRenderer(size: CGSize(width: 4032, height: 3024), format: format).image { _ in
            cat.draw(in: CGRect(x: 0, y: 0, width: 4032, height: 3024))
        }
        let jpeg = folder.appendingPathComponent("photo.jpg")
        try #require(photo.jpegData(compressionQuality: 0.9)).write(to: jpeg)
        let heic = folder.appendingPathComponent("portrait.heic")
        let destination = try #require(CGImageDestinationCreateWithURL(heic as CFURL, "public.heic" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, try #require(photo.cgImage), [kCGImagePropertyOrientation: 6] as CFDictionary)
        try #require(CGImageDestinationFinalize(destination))
        let originals = try [jpeg, heic].map { try Data(contentsOf: $0) }
        var preparedSizes: [[Int]] = []
        for (index, url) in [jpeg, heic].enumerated() {
            let output = try #require(FileOperationManager.shared.processImageFile(from: url,
                fileName: folder.lastPathComponent + "-\(index).jpg"))
            defer { try? FileManager.default.removeItem(at: output) }
            let prepared = try #require(UIImage(contentsOfFile: output.path))
            let bitmap = try #require(prepared.cgImage)
            #expect(prepared.imageOrientation == .up)
            #expect(max(bitmap.width, bitmap.height) == 1024)
            #expect(min(bitmap.width, bitmap.height) == 768)
            #expect(index == 0 ? bitmap.width > bitmap.height : bitmap.height > bitmap.width)
            #expect(try Data(contentsOf: url) == originals[index])
            preparedSizes.append([bitmap.width, bitmap.height])
        }

        let model = ModelInfo(modelName: "HighResolutionImageQA-" + UUID().uuidString + "-Qwen3.5-0.8B-MNN")
        try prepareUsabilityModel(model, maxNewTokens: 96)
        let manager = ModelConfigManager(modelPath: model.localPath)
        manager.updateEnableThinking(false)
        manager.updateConversationContextEnabled(false)
        manager.updateContextTokenLimit(4096)
        manager.saveUseMultimodalPromptAPI(false)
        let chat = LLMChatViewModel(modelInfo: model, showsModelLoadingMessages: false)
        defer { cleanUsabilityModel(model, chat: chat) }
        chat.onStart()
        try await waitUntil(attempts: 5_000) { chat.isModelLoaded || chat.loadFailed }
        try #require(chat.isModelLoaded)
        let media = Media(source: URLMediaModel(url: jpeg))
        chat.sendToLLM(draft: DraftMessage(text: "这张图片中的主要动物是什么？只回答动物名称。", thinkText: "",
            medias: [media], recording: nil, replyMessage: nil, createdAt: Date()))
        try await waitUntil(attempts: 5_000) { chat.isProcessing }
        try await waitUntil(attempts: 5_000) { !chat.isProcessing }
        let reply = chat.messages.last(where: { $0.user.id == "2" })?.text ?? ""
        #expect(reply.contains("猫") || reply.localizedCaseInsensitiveContains("cat"))
        let attachment = try #require(chat.messages.last(where: { $0.user.isCurrentUser })?.attachments.first)
        let savedOriginal = try #require(CGImageSourceCreateWithURL(attachment.full as CFURL, nil))
        let savedProperties = try #require(CGImageSourceCopyPropertiesAtIndex(savedOriginal, 0, nil) as? [CFString: Any])
        #expect(savedProperties[kCGImagePropertyPixelWidth] as? Int == 4032)
        #expect(savedProperties[kCGImagePropertyPixelHeight] as? Int == 3024)
        chat.regenerateLastResponse()
        try await waitUntil(attempts: 5_000) { chat.isProcessing }
        try await waitUntil(attempts: 5_000) { !chat.isProcessing }
        let retry = chat.messages.last(where: { $0.user.id == "2" })?.text ?? ""
        #expect(retry.contains("猫") || retry.localizedCaseInsensitiveContains("cat"))
        #expect(chat.messages.filter { $0.user.isCurrentUser }.count == 1)
        chat.useMultimodalPromptAPI = true
        chat.sendToLLM(draft: DraftMessage(text: "", thinkText: "", medias: [Media(source: URLMediaModel(url: heic))],
            recording: nil, replyMessage: nil, createdAt: Date()))
        try await waitUntil(attempts: 5_000) { chat.isProcessing }
        try await waitUntil(attempts: 5_000) { !chat.isProcessing }
        let imageOnly = chat.messages.last(where: { $0.user.id == "2" })?.text ?? ""
        #expect(imageOnly.contains("猫") || imageOnly.localizedCaseInsensitiveContains("cat"))
        #expect(chat.messages.filter { $0.user.isCurrentUser }.count == 2)
        try saveUsabilityEvidence("photo-input-high-resolution.json", ["backend": "metal",
            "original_size": [4032, 3024], "prepared_sizes": preparedSizes, "original_history_preserved": true,
            "jpeg_reply": reply, "jpeg_retry": retry, "heic_image_only_reply": imageOnly])
    }

    @Test func imageInputUsesExportedCapabilitiesBeforeModelNames() throws {
        let model = ModelInfo(modelName: "ImageCapabilityQA-" + UUID().uuidString, tags: ["image"])
        let folder = URL(fileURLWithPath: model.localPath)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let metadata = folder.appendingPathComponent("llm_config.json")
        try JSONSerialization.data(withJSONObject: ["is_visual": true]).write(to: metadata)
        #expect(ModelUtils.supportsImageInput(model))
        try JSONSerialization.data(withJSONObject: ["is_visual": false]).write(to: metadata)
        #expect(!ModelUtils.supportsImageInput(model))
        try FileManager.default.removeItem(at: metadata)
        try JSONSerialization.data(withJSONObject: ["is_visual": false]).write(to: folder.appendingPathComponent("config.json"))
        #expect(!ModelUtils.supportsImageInput(model))
        #expect(ModelUtils.supportsImageInput(ModelInfo(modelName: "Qwen2.5-VL-QA")))
        #expect(!ModelUtils.supportsImageInput(ModelInfo(modelName: "TextOnlyQA")))
        #expect(!ModelUtils.supportsImageInput(ModelInfo(modelName: "stable-diffusion-QA", tags: ["image"])))
    }

    @Test func imagePreparationAcceptsScreenshotsAndUprightsCameraPhotos() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("ImagePreparationQA-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let image = UIGraphicsImageRenderer(size: CGSize(width: 64, height: 32), format: format).image { context in
            UIColor.red.setFill(); context.fill(CGRect(x: 0, y: 0, width: 32, height: 32))
            UIColor.blue.setFill(); context.fill(CGRect(x: 32, y: 0, width: 32, height: 32))
        }
        let png = folder.appendingPathComponent("screenshot.png")
        let original = try #require(image.pngData())
        try original.write(to: png)
        let prepared = try #require(FileOperationManager.shared.processImageFile(from: png, fileName: "screenshot.png"))
        #expect(prepared.pathExtension == "png")
        #expect(try Data(contentsOf: prepared) == original)
        let diffusion = try #require(FileOperationManager.shared.processImageFile(from: png, fileName: folder.lastPathComponent + ".jpg", requiresJPEG: true))
        defer { try? FileManager.default.removeItem(at: diffusion) }
        #expect(diffusion.pathExtension == "jpg")
        #expect(UIImage(contentsOfFile: diffusion.path)?.size == image.size)
        // A real JPEG orientation tag reproduces camera/library portrait input.
        let photo = folder.appendingPathComponent("portrait.jpg")
        let destination = try #require(CGImageDestinationCreateWithURL(photo as CFURL, "public.jpeg" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, try #require(image.cgImage), [kCGImagePropertyOrientation: 6] as CFDictionary)
        #expect(CGImageDestinationFinalize(destination))
        #expect(UIImage(contentsOfFile: photo.path)?.imageOrientation == .right)
        let uprightURL = try #require(FileOperationManager.shared.processImageFile(from: photo, fileName: folder.lastPathComponent + "-upright.jpg"))
        defer { try? FileManager.default.removeItem(at: uprightURL) }
        let upright = try #require(UIImage(contentsOfFile: uprightURL.path))
        #expect(upright.imageOrientation == .up)
        #expect(upright.size == CGSize(width: 32, height: 64))
    }

    @Test func selectedPhotosRemainInTheComposerUntilExplicitSend() async throws {
        let input = InputViewModel()
        input.onStart()
        defer { input.onStop() }
        let url = try #require(Bundle.main.resourceURL).appendingPathComponent("LocalModel/preset_images/cat.jpg")
        #expect(FileManager.default.fileExists(atPath: url.path))
        let media = Media(source: URLMediaModel(url: url))
        var sent: DraftMessage?
        input.didSendMessage = { draft in Task { @MainActor in sent = draft } }
        input.inputViewAction()(.photo)
        #expect(input.showPicker && input.mediaPickerMode == .photos)
        input.attachments.medias = [media]
        input.showPicker = false
        try await waitUntil { input.state == .hasTextOrMedia }
        #expect(input.attachments.medias.count == 1)
        #expect(sent == nil)
        let localization = ChatLocalization(inputPlaceholder: "输入消息…", signatureText: "添加说明…", cancelButtonText: "取消",
            recentToggleText: "最近照片", waitingForNetwork: "等待网络", recordingText: "录音中", replyToText: "回复")
        func composer(_ type: AvailableInputType) -> some View {
            VStack { Spacer(); InputView(viewModel: input, inputFieldId: UUID(), style: .message,
                availableInput: type, messageUseMarkdown: true, localization: localization) }
                .environmentObject(GlobalFocusState())
                .chatTheme(colors: .init(mainBG: Color(uiColor: .systemBackground), mainTint: .secondary,
                    mainText: .primary, inputBG: Color(uiColor: .secondarySystemBackground), inputText: .primary,
                    inputPlaceholderText: .secondary, sendButtonBackground: .blue),
                    images: .init(arrowSend: Image(systemName: "arrow.up")))
        }
        try await capture("photo-composer-preview", view: composer(.textAndMedia))
        input.attachments.medias = []
        try await waitUntil { input.state == .empty }
        try await capture("photo-composer-empty", view: composer(.textAndMedia))
        try await capture("text-only-composer", view: composer(.textOnly))
        input.inputViewAction()(.camera)
        #expect(input.showPicker && input.mediaPickerMode == .camera)
        input.attachments.medias = [media]
        input.showPicker = false
        try await waitUntil { input.state == .hasTextOrMedia }
        #expect(input.attachments.medias.count == 1 && sent == nil)
        input.text = "这张照片里是什么？"
        input.inputViewAction()(.send)
        try await waitUntil { sent != nil && input.attachments.medias.isEmpty }
        let draft = try #require(sent)
        #expect(draft.text == "这张照片里是什么？")
        #expect(draft.medias.count == 1)
        #expect(await draft.medias.first?.getURL() == url)
    }

    @Test func metalImageUploadSupportsPNGAndRegeneration() async throws {
        try await waitUntil(attempts: 500) { UIApplication.shared.applicationState == .active }
        let model = ModelInfo(modelName: "ImageInputQA-" + UUID().uuidString + "-Qwen3.5-0.8B-MNN")
        try prepareUsabilityModel(model, maxNewTokens: 96)
        let manager = ModelConfigManager(modelPath: model.localPath)
        manager.updateEnableThinking(false)
        manager.updateConversationContextEnabled(false)
        manager.updateContextTokenLimit(4096)
        manager.saveUseMultimodalPromptAPI(false)
        let chat = LLMChatViewModel(modelInfo: model, showsModelLoadingMessages: false)
        defer { cleanUsabilityModel(model, chat: chat) }
        #expect(chat.supportsImageInput) // No VL name or image catalog tag.
        chat.onStart()
        try await waitUntil(attempts: 5_000) { chat.isModelLoaded || chat.loadFailed }
        #expect(chat.isModelLoaded)
        guard chat.isModelLoaded else { return }
        let originalURL = try #require(Bundle.main.resourceURL).appendingPathComponent("LocalModel/preset_images/cat.jpg")
        let originalImage = try #require(UIImage(contentsOfFile: originalURL.path))
        let png = FileManager.default.temporaryDirectory.appendingPathComponent("ImageUploadQA-" + UUID().uuidString + ".png")
        try #require(originalImage.pngData()).write(to: png)
        defer { try? FileManager.default.removeItem(at: png) }
        let media = Media(source: URLMediaModel(url: png))
        chat.sendToLLM(draft: DraftMessage(text: "这张图片中的主要动物是什么？只回答动物名称。", thinkText: "",
            medias: [media], recording: nil, replyMessage: nil, createdAt: Date()))
        try await waitUntil(attempts: 5_000) { !chat.isProcessing }
        let reply = try #require(chat.messages.last(where: { $0.user.id == "2" }))
        #expect(reply.text.contains("猫") || reply.text.localizedCaseInsensitiveContains("cat"))
        #expect(chat.messages.last(where: { $0.user.isCurrentUser })?.attachments.count == 1)
        let saved = try #require(chat.savedConversation())
        #expect(saved.messages.first(where: { $0.isUser })?.images?.count == 1)
        let controls = ChatWorkspaceControls(models: [model], onSelectModel: { _ in }, onManageModels: {},
            onHistory: {}, onNewConversation: {})
        try await capture("photo-chat-metal", view: LLMChatView(viewModel: chat, draftText: .constant(""), controls: controls))
        chat.regenerateLastResponse()
        try await waitUntil(attempts: 5_000) { !chat.isProcessing }
        let retry = chat.messages.last(where: { $0.user.id == "2" })?.text ?? ""
        #expect(retry.contains("猫") || retry.localizedCaseInsensitiveContains("cat"))
        #expect(chat.messages.filter { $0.user.isCurrentUser }.count == 1)
        chat.useMultimodalPromptAPI = true
        chat.sendToLLM(draft: DraftMessage(text: "", thinkText: "", medias: [media],
            recording: nil, replyMessage: nil, createdAt: Date()))
        try await waitUntil(attempts: 5_000) { !chat.isProcessing }
        let imageOnly = chat.messages.last(where: { $0.user.id == "2" })?.text ?? ""
        #expect(!imageOnly.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!imageOnly.contains("Error"))
        #expect(chat.messages.filter { $0.user.isCurrentUser }.count == 2)
        try saveUsabilityEvidence("photo-input-metal.json", ["backend": "metal", "input_format": "png",
            "reply": reply.text, "retry": retry, "image_only": imageOnly, "history_image_count": saved.messages.first(where: { $0.isUser })?.images?.count ?? 0])
    }

    @Test func legacyChatMLTemplatesKeepTheirPrefixAndPreserveModernJinja() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("ChatMLRegression-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let file = folder.appendingPathComponent("llm_config.json")
        let prefix = "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
        let legacy = prefix + "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
        try JSONSerialization.data(withJSONObject: ["prompt_template": legacy, "hidden_size": 960]).write(to: file)
        let manager = ModelConfigManager(modelPath: folder.path)
        try manager.prepareModelForInference()
        let repaired = try Data(contentsOf: file)
        let config = try #require(JSONSerialization.jsonObject(with: repaired) as? [String: Any])
        let jinja = try #require(config["jinja"] as? [String: Any])
        #expect((jinja["context"] as? [String: String])?["legacy_prefix"] == prefix)
        #expect(jinja["eos"] as? String == "<|im_end|>")
        #expect(config["prompt_template"] as? String == legacy)
        #expect(config["hidden_size"] as? Int == 960)
        try manager.prepareModelForInference()
        #expect(try Data(contentsOf: file) == repaired)
        let modern: [String: Any] = ["prompt_template": legacy, "jinja": ["chat_template": "custom", "context": ["enable_thinking": false]]]
        let original = try JSONSerialization.data(withJSONObject: modern)
        try original.write(to: file)
        try manager.prepareModelForInference()
        #expect(try Data(contentsOf: file) == original)
        let other = try JSONSerialization.data(withJSONObject: ["prompt_template": "<s>[INST] %s [/INST]"])
        try other.write(to: file)
        try manager.prepareModelForInference()
        #expect(try Data(contentsOf: file) == other)
    }

    @Test func smollm2LegacyTemplateSupportsMetalChatAndConversationContext() async throws {
        try await waitUntil(attempts: 500) { UIApplication.shared.applicationState == .active }
        let original = ModelInfo(modelName: "SmolLM2-360M-Instruct-MNN")
        let source = URL(fileURLWithPath: original.localPath)
        let clone = ModelInfo(modelName: "SmolLMRegression-" + UUID().uuidString)
        let folder = URL(fileURLWithPath: clone.localPath)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        for file in try FileManager.default.contentsOfDirectory(at: source, includingPropertiesForKeys: nil)
            where !["config.json", "llm_config.json"].contains(file.lastPathComponent) {
            try FileManager.default.createSymbolicLink(at: folder.appendingPathComponent(file.lastPathComponent), withDestinationURL: file)
        }
        var metadata = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: source.appendingPathComponent("llm_config.json"))) as? [String: Any])
        metadata.removeValue(forKey: "jinja") // Exercise the published legacy package even after migration.
        try JSONSerialization.data(withJSONObject: metadata).write(to: folder.appendingPathComponent("llm_config.json"))
        var config = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: source.appendingPathComponent("config.json"))) as? [String: Any])
        config["backend_type"] = "metal"
        config["max_new_tokens"] = 64
        config["sampler_type"] = "greedy"
        try JSONSerialization.data(withJSONObject: config).write(to: folder.appendingPathComponent("config.json"))
        try ModelConfigManager(modelPath: folder.path).prepareModelForInference()
        let state = MetalProbeState()
        let engine = LLMInferenceEngineWrapper(modelPath: folder.path) { success in
            Task { @MainActor in state.loaded = success }
        }
        defer { engine.cancelInference() }
        try await waitUntil(attempts: 5_000) { state.loaded != nil }
        #expect(state.loaded == true)
        guard state.loaded == true else { return }
        engine.setContextTokenLimit(512)
        engine.setConversationContextEnabled(false)
        func ask(_ prompt: String) async throws -> String {
            try await waitUntil { !engine.isProcessing() }
            state.reply = ""
            state.finished = false
            engine.processInput(prompt, withOutput: { output in
                Task { @MainActor in
                    if output.contains("<eop>") || output.contains("<stoped>") { state.finished = true }
                    else { state.reply += output }
                }
            })
            try await waitUntil(attempts: 3_000) { state.finished }
            return state.reply
        }
        let first = try await ask("What is the capital of France?")
        #expect(first.lowercased().contains("paris"))
        #expect(engine.contextUsage()["inputTokens"]?.intValue == 37)
        engine.setConversationContextEnabled(true)
        engine.addPrompts(from: [["user": "My name is Alice."], ["assistant": "Hello Alice."]])
        let recalled = try await ask("What is my name? Reply with just the name.")
        #expect(recalled.lowercased().contains("alice"))
        try saveUsabilityEvidence("smollm2-template-regression.json", ["backend": "metal", "first_reply": first,
                                                                   "context_reply": recalled])
    }

    @Test func legacyMiniCPMTokenizerRepairPreservesLiteralSpaceAndExplicitFlags() throws {
        let model = ModelInfo(modelName: "TokenizerRepairQA-" + UUID().uuidString + "-MiniCPM4-0.5B-MNN")
        let folder = URL(fileURLWithPath: model.localPath)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let file = folder.appendingPathComponent("tokenizer.txt")
        let old = "430 0\n0 0 0\n\n4\nIGE= 1 1\nIA== 0 1\nIA== -9 1\nYQ== 0 1\n"
        try Data(old.utf8).write(to: file)
        let manager = ModelConfigManager(modelPath: folder.path)
        try manager.prepareTokenizerForInference()
        let repaired = try String(contentsOf: file, encoding: .utf8)
        #expect(repaired.contains("\n4 0 1\n"))
        #expect(repaired.contains(Data("▁a".utf8).base64EncodedString() + " 1 1"))
        #expect(repaired.contains("IA== 0 1"))
        #expect(repaired.contains(Data("▁".utf8).base64EncodedString() + " -9 1"))
        try manager.prepareTokenizerForInference()
        #expect(try String(contentsOf: file, encoding: .utf8) == repaired)
        let explicit = old.replacingOccurrences(of: "\n4\n", with: "\n4 1 0\n")
        try Data(explicit.utf8).write(to: file)
        try manager.prepareTokenizerForInference()
        #expect(try String(contentsOf: file, encoding: .utf8) == explicit)
        let configFile = folder.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: ["backend_type": "cpu", "temperature": 0.7]).write(to: configFile)
        try manager.prepareModelForInference()
        #expect(manager.readBackendType() == "metal")
        manager.updateBackendType("cpu")
        try manager.prepareModelForInference()
        #expect(manager.readBackendType() == "cpu")
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try? FileManager.default.removeItem(at: documents.appendingPathComponent("MNNConfigs/" + model.modelName))
    }

    @Test func completedDownloadsDefaultToMetalAndKeepUserOverrides() async throws {
        let model = ModelInfo(modelName: "MetalDownloadQA-" + UUID().uuidString)
        let folder = URL(fileURLWithPath: model.localPath)
        let configFile = folder.appendingPathComponent("config.json")
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let customFolder = documents.appendingPathComponent("MNNConfigs/" + model.modelName)
        let storage = ModelStorageManager.shared
        let originalDownloads = storage.downloadedModels
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let serverConfig: [String: Any] = ["backend_type": "cpu", "thread_num": 4, "temperature": 0.7,
                                         "mllm": ["backend_type": "cpu"], "llm_model": "llm.mnn"]
        try JSONSerialization.data(withJSONObject: serverConfig).write(to: configFile)
        defer {
            storage.downloadedModels = originalDownloads
            storage.clearCachedSize(for: model.modelName)
            try? FileManager.default.removeItem(at: folder)
            try? FileManager.default.removeItem(at: customFolder)
        }
        let client = ControlledModelClient()
        let listing = ModelListViewModel(modelClient: client, fetchOnInit: false)
        listing.models = [model]
        var observedBackend: String?
        let observation = listing.$models.sink { models in
            if models.first?.isDownloaded == true {
                observedBackend = ModelConfigManager(modelPath: model.localPath).readBackendType()
            }
        }
        defer { observation.cancel() }
        let download = Task { await listing.downloadModel(model) }
        try await waitUntil { client.downloads.count == 1 }
        client.downloads[0].completion.resume()
        await download.value
        #expect(listing.models.first?.isDownloaded == true)
        #expect(observedBackend == "metal")
        let config = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: configFile)) as? [String: Any])
        #expect(config["backend_type"] as? String == "metal")
        #expect(config["temperature"] as? Double == 0.7)
        #expect((config["mllm"] as? [String: String])?["backend_type"] == "cpu")
        let manager = ModelConfigManager(modelPath: model.localPath)
        manager.updateBackendType("cpu") // An explicit choice survives another download.
        try JSONSerialization.data(withJSONObject: serverConfig).write(to: configFile)
        let redownload = Task { await listing.downloadModel(model) }
        try await waitUntil { client.downloads.count == 2 }
        client.downloads[1].completion.resume()
        await redownload.value
        let defaults = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: configFile)) as? [String: Any])
        #expect(defaults["backend_type"] as? String == "metal")
        #expect(manager.readBackendType() == "cpu")
        let mergedString = try #require(manager.readConfigAsJSONString())
        let merged = try #require(JSONSerialization.jsonObject(with: Data(mergedString.utf8)) as? [String: Any])
        #expect(merged["backend_type"] as? String == "cpu")
    }


    @Test func minicpmMetalSurvivesQwenSwitchAndNewConversation() async throws {
        try await waitUntil(attempts: 500) { UIApplication.shared.applicationState == .active }
        let suffix = UUID().uuidString
        let preferences = try #require(UserDefaults(suiteName: "MiniCPMProbe-" + suffix))
        var workspace = ChatWorkspaceModel(preferences: preferences)
        var models: [ModelInfo] = []
        var evidence: [[String: Any]] = []
        defer {
            cleanWorkspaceModels(models, workspace: workspace)
            preferences.removePersistentDomain(forName: "MiniCPMProbe-" + suffix)
        }
        func save(_ phase: String) throws {
            try saveUsabilityEvidence("minicpm-metal-switching.json", ["phase": phase, "results": evidence])
        }
        func clone(_ original: String, label: String, backend: String? = nil, precision: String? = nil, mmap: Bool = true, extra: [String: Any] = [:]) throws -> ModelInfo {
            let model = ModelInfo(modelName: "MiniCPMProbe-" + label + "-" + suffix + "-" + original, isDownloaded: true)
            models.append(model)
            let source = URL(fileURLWithPath: ModelInfo(modelName: original).localPath)
            let target = URL(fileURLWithPath: model.localPath)
            try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
            var config = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: source.appendingPathComponent("config.json"))) as? [String: Any])
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let originalSettings = documents.appendingPathComponent("MNNConfigs/" + original + "/custom_config.json")
            if let data = try? Data(contentsOf: originalSettings), let settings = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                config.merge(settings) { _, new in new }
            }
            if let backend { config["backend_type"] = backend }
            if let precision { config["precision"] = precision }
            config["max_new_tokens"] = 64
            config["sampler_type"] = "greedy"
            config.merge(extra) { _, new in new }
            config["use_mmap"] = mmap
            config["context_token_limit"] = 4096
            config["conversation_context_enabled"] = true
            for file in try FileManager.default.contentsOfDirectory(at: source, includingPropertiesForKeys: nil)
                where file.lastPathComponent != "config.json" {
                if file.lastPathComponent == "tokenizer.txt" {
                    try FileManager.default.copyItem(at: file, to: target.appendingPathComponent(file.lastPathComponent))
                } else {
                    try FileManager.default.createSymbolicLink(at: target.appendingPathComponent(file.lastPathComponent), withDestinationURL: file)
                }
            }
            try JSONSerialization.data(withJSONObject: config).write(to: target.appendingPathComponent("config.json"))
            ModelConfigManager(modelPath: model.localPath).updateEnableThinking(false)
            return model
        }
        func ready(_ model: ModelInfo) async throws -> LLMChatViewModel {
            workspace.updateModels(models)
            #expect(workspace.selectModel(model))
            let chat = try #require(workspace.currentChat)
            try await waitUntil(attempts: 5_000) { chat.isModelLoaded || chat.loadFailed }
            #expect(chat.isModelLoaded)
            return chat
        }
        func ask(_ chat: LLMChatViewModel, caseName: String, prompt: String) async throws {
            print("[MiniCPMProbe] BEGIN " + caseName)
            try save("running " + caseName)
            chat.sendUserPrompt(prompt)
            try await waitUntil(attempts: 5_000) { !chat.isProcessing }
            let message = try #require(chat.messages.last(where: { $0.user.id == "2" }))
            evidence.append(["case": caseName, "backend": chat.modelConfigManager.readBackendType(),
                             "precision": chat.modelConfigManager.readPrecision(), "mmap": chat.modelConfigManager.readUseMmap(), "prompt": prompt,
                             "reply": message.text, "thinking_characters": message.thinkText?.count ?? 0,
                             "input_tokens": chat.contextUsage.inputTokens, "generated_tokens": chat.contextUsage.generatedTokens,
                             "history_user_messages": chat.messages.filter { $0.user.isCurrentUser }.count])
            #expect(!message.text.contains("▁"))
            #expect(!message.text.contains("<0x"))
            #expect(!message.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            try save("completed " + caseName)
        }
        let mini = try clone("MiniCPM4-0.5B-MNN", label: "metal-low", backend: "metal", precision: "low")
        _ = try await ready(mini)
        for iteration in 1...3 {
            #expect(workspace.newConversation())
            try await ask(try #require(workspace.currentChat), caseName: "fresh-mini-metal-\(iteration)",
                          prompt: "1+1等于几？只回答数字。")
            #expect(workspace.currentChat?.messages.last(where: { $0.user.id == "2" })?.text.contains("2") == true)
        }
        #expect(workspace.newConversation())
        let high = try clone("MiniCPM4-0.5B-MNN", label: "metal-high", backend: "metal", precision: "high")
        _ = try await ready(high)
        try await ask(try #require(workspace.currentChat), caseName: "mini-metal-high", prompt: "1+1等于几？只回答数字。")
        #expect(workspace.currentChat?.messages.last(where: { $0.user.id == "2" })?.text.contains("2") == true)
        #expect(workspace.newConversation())
        let qwen = try clone("Qwen3.5-0.8B-MNN", label: "qwen-metal", backend: "metal", precision: "low")
        _ = try await ready(qwen)
        try await ask(try #require(workspace.currentChat), caseName: "qwen-before-switch", prompt: "请记住数字7，只回答7。")
        let conversation = workspace.activeHistoryID
        _ = try await ready(mini) // The old chat is released inside selectModel, with no extra strong reference.
        #expect(workspace.activeHistoryID == conversation)
        try await ask(try #require(workspace.currentChat), caseName: "mini-after-qwen-with-context", prompt: "刚才让你记住的数字是什么？")
        #expect(workspace.currentChat?.messages.last(where: { $0.user.id == "2" })?.text.contains("7") == true)
        #expect(workspace.newConversation())
        try await ask(try #require(workspace.currentChat), caseName: "mini-new-chat-after-switch", prompt: "1+1等于几？只回答数字。")
        #expect(workspace.currentChat?.messages.last(where: { $0.user.id == "2" })?.text.contains("2") == true)
        _ = try await ready(qwen)
        try await ask(try #require(workspace.currentChat), caseName: "qwen-after-mini", prompt: "1+1等于几？只回答数字。")
        #expect(workspace.currentChat?.messages.last(where: { $0.user.id == "2" })?.text.contains("2") == true)
        try save("complete")
    }

    @Test func metalModelProducesAShortReply() async throws {
        try await waitUntil(attempts: 500) { UIApplication.shared.applicationState == .active }
        let model = ModelInfo(modelId: "taobao-mnn/Qwen3.5-0.8B-MNN")
        let source = URL(fileURLWithPath: model.localPath)
        let originalConfig = source.appendingPathComponent("config.json")
        #expect(FileManager.default.fileExists(atPath: originalConfig.path))
        var config = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: originalConfig)) as? [String: Any])
        config["backend_type"] = "metal"
        config["max_new_tokens"] = 32
        config["sampler_type"] = "greedy"
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("ReleaseQA-" + UUID().uuidString + "-Qwen3.5-0.8B-MNN")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let cacheName = (folder.lastPathComponent as NSString).deletingPathExtension
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("MNN_" + cacheName + "_temp")
        defer {
            try? FileManager.default.removeItem(at: folder)
            try? FileManager.default.removeItem(at: cache)
        }
        for file in try FileManager.default.contentsOfDirectory(at: source, includingPropertiesForKeys: nil) where file.lastPathComponent != "config.json" {
            try FileManager.default.createSymbolicLink(at: folder.appendingPathComponent(file.lastPathComponent), withDestinationURL: file)
        }
        try JSONSerialization.data(withJSONObject: config).write(to: folder.appendingPathComponent("config.json"))
        let state = MetalProbeState()
        let engine = LLMInferenceEngineWrapper(modelPath: folder.path) { success in
            Task { @MainActor in state.loaded = success }
        }
        defer { engine.cancelInference() }
        try await waitUntil(attempts: 5_000) { state.loaded != nil }
        #expect(state.loaded == true)
        guard state.loaded == true else { return }
        engine.setContextTokenLimit(512)
        engine.setThinkingModeEnabled(false)
        engine.processInput("请只回答 OK。", withOutput: { output in
            Task { @MainActor in
                if output.contains("<eop>") || output.contains("<stoped>") {
                    state.finished = true
                } else {
                    state.reply += output
                }
            }
        })
        try await waitUntil(attempts: 3_000) { state.finished }
        #expect(!state.reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!state.reply.contains("Error"))
        let result: [String: Any] = ["model": model.modelName, "requested_backend": "metal", "finished": state.finished, "reply": state.reply]
        let directory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("ReleaseQA")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: result, options: .prettyPrinted).write(to: directory.appendingPathComponent("metal-smoke.json"))

        func ask(_ text: String) async throws -> String {
            try await waitUntil { !engine.isProcessing() }
            state.finished = false
            state.reply = ""
            engine.processInput(text, withOutput: { output in
                Task { @MainActor in
                    if output.contains("<eop>") || output.contains("<stoped>") { state.finished = true }
                    else { state.reply += output }
                }
            })
            try await waitUntil(attempts: 3_000) { state.finished }
            return state.reply
        }
        engine.setConversationContextEnabled(true)
        engine.addPrompts(from: [["user": "我的暗号是松果73。请记住它。"], ["assistant": "记住了。"]])
        let remembered = try await ask("我的暗号是什么？只回答暗号。")
        #expect(remembered.filter { !$0.isWhitespace }.contains("松果73"))
        #expect(engine.getChatHistoryCount() == 3)
        engine.setConversationContextEnabled(false)
        let independent = try await ask("我的暗号是什么？只回答暗号。")
        #expect(engine.getChatHistoryCount() == 1)
        engine.clearChatHistory()
        engine.setConversationContextEnabled(true)
        let freshConversation = try await ask("请只回答 OK。")
        #expect(engine.getChatHistoryCount() == 1)
        let beforeOversized = engine.contextUsage()
        #expect((beforeOversized["inputTokens"]?.intValue ?? 0) + (beforeOversized["generatedTokens"]?.intValue ?? 0) <= 512)
        try await waitUntil { !engine.isProcessing() }
        // Replacing a prompt set is preparation, not a user-visible reset.
        engine.addPrompts(from: [])
        #expect(engine.getChatHistoryCount() == 0)
        #expect(engine.contextUsage()["inputTokens"] == beforeOversized["inputTokens"])
        #expect(engine.contextUsage()["generatedTokens"] == beforeOversized["generatedTokens"])
        engine.addPrompts(from: [["user": "新一轮历史"], ["assistant": "已确认"]])
        #expect(engine.contextUsage()["inputTokens"] == beforeOversized["inputTokens"])
        #expect(engine.contextUsage()["generatedTokens"] == beforeOversized["generatedTokens"])
        engine.clearChatHistory()
        #expect(engine.contextUsage()["inputTokens"]?.intValue == 0)
        #expect(engine.contextUsage()["generatedTokens"]?.intValue == 0)
        engine.setThinkingModeEnabled(true)
        engine.setConfigWithJSONString("{\"max_new_tokens\":4096}")
        let longRequest = "请尽可能详细地思考，并逐一列出1到1000的数字，解释每个数字，不要省略。"
        let expectedInput = await withCheckedContinuation { continuation in
            engine.countTokens(for: [["user": longRequest]]) { continuation.resume(returning: $0) }
        }
        let budgetStoppedReply = try await ask(longRequest)
        let bounded = engine.contextUsage()
        #expect(bounded["inputTokens"]?.intValue == expectedInput)
        #expect((bounded["generatedTokens"]?.intValue ?? 0) > 128)
        #expect((bounded["inputTokens"]?.intValue ?? 0) + (bounded["generatedTokens"]?.intValue ?? 0) <= 512)
        #expect(bounded["budgetExhausted"]?.boolValue == true)
        #expect(!budgetStoppedReply.isEmpty)
        let generationResult: [String: Any] = ["context_limit": 512, "requested_max_new_tokens": 4096,
            "thinking": true, "input_tokens": bounded["inputTokens"]?.intValue ?? -1,
            "generated_tokens": bounded["generatedTokens"]?.intValue ?? -1, "planned_input_tokens": expectedInput]
        try JSONSerialization.data(withJSONObject: generationResult, options: .prettyPrinted)
            .write(to: directory.appendingPathComponent("context-generation-limit.json"))
        engine.clearChatHistory()
        let oversizedReply = try await ask(String(repeating: "请仔细检查这段测试内容。", count: 100))
        #expect(oversizedReply.contains("Input exceeds the context budget"))
        #expect(engine.contextUsage()["generatedTokens"]?.intValue == 0)
        let contextResult: [String: Any] = ["backend": "metal", "enabled_reply": remembered,
            "disabled_reply": independent, "new_conversation_reply": freshConversation,
            "enabled_prompt_messages": 3, "disabled_prompt_messages": 1, "new_conversation_prompt_messages": 1]
        try JSONSerialization.data(withJSONObject: contextResult, options: .prettyPrinted)
            .write(to: directory.appendingPathComponent("conversation-context.json"))
    }

    @Test func metalConversationAutomaticallySummarizesAndPreservesRawHistory() async throws {
        try await waitUntil(attempts: 500) { UIApplication.shared.applicationState == .active }
        let source = URL(fileURLWithPath: ModelInfo(modelId: "taobao-mnn/Qwen3.5-0.8B-MNN").localPath)
        let model = ModelInfo(modelName: "ContextBudgetQA-" + UUID().uuidString + "-Qwen3.5-0.8B-MNN", tags: ["Think"])
        let folder = URL(fileURLWithPath: model.localPath)
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let custom = documents.appendingPathComponent("MNNConfigs/" + model.modelName)
        let historyId = UUID().uuidString
        let historyFolder = documents.appendingPathComponent("ChatHistories/" + historyId)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for file in try FileManager.default.contentsOfDirectory(at: source, includingPropertiesForKeys: nil) where file.lastPathComponent != "config.json" {
            try FileManager.default.createSymbolicLink(at: folder.appendingPathComponent(file.lastPathComponent), withDestinationURL: file)
        }
        var config = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: source.appendingPathComponent("config.json"))) as? [String: Any])
        config["backend_type"] = "metal"
        config["sampler_type"] = "greedy"
        config["max_new_tokens"] = 128
        try JSONSerialization.data(withJSONObject: config).write(to: folder.appendingPathComponent("config.json"))
        let manager = ModelConfigManager(modelPath: folder.path)
        manager.updateEnableThinking(false)
        manager.updateConversationContextEnabled(true)
        manager.updateContextTokenLimit(1024)
        let filler = String(repeating: "Background test note; it adds no new facts. ", count: 10)
        let rawMessages = (0..<6).flatMap { index -> [HistoryMessage] in
            let user = index == 0 ? "我的暗号是松果73，请一直记住。\n" + filler : index < 4 ? filler : "第\(index)轮确认。"
            let assistant = index == 0 ? "已记住你的暗号是松果73。\n" + filler : index < 4 ? filler : "确认完成。"
            return [HistoryMessage(id: UUID().uuidString, content: user, images: nil, audio: nil, isUser: true, createdAt: Date()),
                    HistoryMessage(id: UUID().uuidString, content: assistant, images: nil, audio: nil, isUser: false, createdAt: Date())]
        }
        let history = ChatHistory(id: historyId, modelInfo: model, messages: rawMessages, createdAt: Date(), updatedAt: Date())
        let viewModel = LLMChatViewModel(modelInfo: model, history: history)
        defer {
            viewModel.onStop()
            for saved in ChatHistoryManager.shared.getAllHistory().filter({ $0.modelId == model.id }) {
                ChatHistoryManager.shared.deleteHistory(saved)
            }
            try? FileManager.default.removeItem(at: historyFolder)
            try? FileManager.default.removeItem(at: custom)
            try? FileManager.default.removeItem(at: folder)
        }
        viewModel.onStart()
        try await waitUntil(attempts: 5_000) { viewModel.isModelLoaded || viewModel.loadFailed }
        #expect(viewModel.isModelLoaded)
        guard viewModel.isModelLoaded else { return }
        func askForSecret() {
            viewModel.sendToLLM(draft: DraftMessage(text: "我的暗号是什么？只回答暗号。", thinkText: "", medias: [],
                                                  recording: nil, replyMessage: nil, createdAt: Date()))
        }
        askForSecret()
        try await waitUntil(attempts: 2_000) { viewModel.isCompactingContext }
        viewModel.cancelGeneration()
        try await waitUntil(attempts: 3_000) { !viewModel.isProcessing }
        #expect(!viewModel.isCompactingContext)
        #expect(!viewModel.chatInputUnavilable)
        askForSecret()
        try await waitUntil { viewModel.isProcessing }
        try await waitUntil(attempts: 6_000) { !viewModel.isProcessing }
        try await Task.sleep(for: .milliseconds(450))
        let snapshot = try #require(ContextSummarySnapshot.load(historyId: historyId, modelId: model.id))
        // The request cancelled during compaction is also retained as a turn.
        #expect(snapshot.coveredTurns == 5)
        #expect(snapshot.summary.filter { !$0.isWhitespace }.contains("松果73"))
        let reply = viewModel.messages.last(where: { $0.user.id == "2" })?.text ?? ""
        #expect(reply.filter { !$0.isWhitespace }.contains("松果73"))
        #expect(viewModel.contextUsage.inputTokens > 0)
        #expect(viewModel.contextUsage.generatedTokens > 0)
        #expect(viewModel.contextUsage.totalTokens <= 1024)
        #expect(ConversationContextBuilder.turns(from: viewModel.messages).count == 8)
        #expect(viewModel.messages.contains(where: { $0.text == rawMessages[0].content }))
        let directory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("ReleaseQA")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let result: [String: Any] = ["backend": "metal", "context_limit": 1024, "input_tokens": viewModel.contextUsage.inputTokens,
            "generated_tokens": viewModel.contextUsage.generatedTokens, "summary": snapshot.summary, "covered_turns": snapshot.coveredTurns,
            "reply": reply, "notice": viewModel.contextNotice ?? "", "raw_turns": ConversationContextBuilder.turns(from: viewModel.messages).count,
            "cancel_during_compaction_recovered": true, "available_memory_bytes": viewModel.contextUsage.availableMemoryBytes ?? 0,
            "footprint_bytes": viewModel.contextUsage.footprintBytes ?? 0]
        try JSONSerialization.data(withJSONObject: result, options: .prettyPrinted)
            .write(to: directory.appendingPathComponent("context-budget.json"))
        try await capture("context-progress-zh", view: VStack {
            ContextUsageView(viewModel: viewModel)
            Text(reply).padding()
            Spacer()
        })
        try await capture("context-settings-loaded-zh", view: ModelSettingsView(showSettings: .constant(true), viewModel: viewModel))
        // Reject a single oversized message before changing an existing summary.
        viewModel.sendToLLM(draft: DraftMessage(text: String(repeating: "请检查这段超长测试内容。", count: 200), thinkText: "", medias: [],
                                              recording: nil, replyMessage: nil, createdAt: Date()))
        try await waitUntil(attempts: 500) { viewModel.messages.last?.text == AppText.text("context.error.tooLong") }
        #expect(ContextSummarySnapshot.load(historyId: historyId, modelId: model.id) == snapshot)
        #expect(!viewModel.isProcessing)

        let previousTotal = viewModel.contextUsage.totalTokens
        #expect(previousTotal > 0)
        var samples: [Int] = []
        let observation = viewModel.$contextUsage.sink { samples.append($0.totalTokens) }
        defer { observation.cancel() }
        // Poll more frequently than the production timer to expose transient
        // counters during history preparation and native prefill.
        let polling = Task { @MainActor in
            while !Task.isCancelled {
                viewModel.refreshContextUsage()
                try? await Task.sleep(for: .milliseconds(2))
            }
        }
        defer { polling.cancel() }
        for _ in 0..<3 {
            askForSecret()
            try await waitUntil { viewModel.isProcessing }
            try await waitUntil(attempts: 3_000) { !viewModel.isProcessing }
        }
        polling.cancel()
        #expect(!samples.isEmpty)
        #expect(samples.allSatisfy { $0 > 0 })
        #expect(viewModel.messages.last(where: { $0.user.id == "2" })?.text.filter { !$0.isWhitespace }.contains("松果73") == true)
        let progressResult: [String: Any] = ["backend": "metal", "previous_total": previousTotal, "request_count": 3,
            "sample_count": samples.count, "samples": samples, "zero_sample_count": samples.filter { $0 == 0 }.count]
        try JSONSerialization.data(withJSONObject: progressResult, options: .prettyPrinted)
            .write(to: directory.appendingPathComponent("context-progress-continuity.json"))
        viewModel.resetConversation()
        viewModel.refreshContextUsage()
        #expect(viewModel.contextUsage.totalTokens == 0)
    }

    @Test func metalConversationRetainsRequestsWithoutFinalAnswers() async throws {
        try await waitUntil(attempts: 500) { UIApplication.shared.applicationState == .active }
        let source = URL(fileURLWithPath: ModelInfo(modelId: "taobao-mnn/Qwen3.5-0.8B-MNN").localPath)
        let model = ModelInfo(modelName: "ContextRetentionQA-" + UUID().uuidString + "-Qwen3.5-0.8B-MNN", tags: ["Think"])
        let folder = URL(fileURLWithPath: model.localPath)
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let custom = documents.appendingPathComponent("MNNConfigs/" + model.modelName)
        let historyId = UUID().uuidString
        let historyFolder = documents.appendingPathComponent("ChatHistories/" + historyId)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for file in try FileManager.default.contentsOfDirectory(at: source, includingPropertiesForKeys: nil) where file.lastPathComponent != "config.json" {
            try FileManager.default.createSymbolicLink(at: folder.appendingPathComponent(file.lastPathComponent), withDestinationURL: file)
        }
        var config = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: source.appendingPathComponent("config.json"))) as? [String: Any])
        config["backend_type"] = "metal"
        config["sampler_type"] = "greedy"
        // Force an actual thinking-only token cutoff deterministically, without
        // spending thousands of tokens to reach the same empty-final-answer path.
        config["max_new_tokens"] = 8
        try JSONSerialization.data(withJSONObject: config).write(to: folder.appendingPathComponent("config.json"))
        let manager = ModelConfigManager(modelPath: folder.path)
        manager.updateEnableThinking(true)
        manager.updateConversationContextEnabled(true)
        manager.updateContextTokenLimit(4096)
        let history = ChatHistory(id: historyId, modelInfo: model, messages: [], createdAt: Date(), updatedAt: Date())
        let viewModel = LLMChatViewModel(modelInfo: model, history: history)
        defer {
            viewModel.onStop()
            for saved in ChatHistoryManager.shared.getAllHistory().filter({ $0.modelId == model.id }) {
                ChatHistoryManager.shared.deleteHistory(saved)
            }
            try? FileManager.default.removeItem(at: historyFolder)
            try? FileManager.default.removeItem(at: custom)
            try? FileManager.default.removeItem(at: folder)
        }
        viewModel.onStart()
        try await waitUntil(attempts: 5_000) { viewModel.isModelLoaded || viewModel.loadFailed }
        #expect(viewModel.isModelLoaded)
        guard viewModel.isModelLoaded else { return }
        func ask(_ prompt: String) async throws {
            viewModel.sendToLLM(draft: DraftMessage(text: prompt, thinkText: "", medias: [],
                                                  recording: nil, replyMessage: nil, createdAt: Date()))
            try await waitUntil { viewModel.isProcessing }
            try await waitUntil(attempts: 3_000) { !viewModel.isProcessing }
        }
        try await ask("我的暗号是松果73。请先仔细推理记忆方法，再确认这个暗号。")
        let firstReply = try #require(viewModel.messages.last(where: { $0.user.id == "2" }))
        #expect(firstReply.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!(firstReply.thinkText ?? "").isEmpty)
        #expect(viewModel.contextUsage.generatedTokens == 8)
        #expect(viewModel.contextNotice == AppText.text("context.noFinalResponse"))
        let firstUsage = viewModel.contextUsage
        viewModel.toggleThinkingMode()
        #expect(!viewModel.isThinkingModeEnabled)
        try await ask("刚才告诉你的暗号是什么？只回答暗号。")
        let reply = viewModel.messages.last(where: { $0.user.id == "2" })?.text ?? ""
        #expect(reply.filter { !$0.isWhitespace }.contains("松果73"))
        #expect(ConversationContextBuilder.turns(from: viewModel.messages).count == 2)
        let directory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("ReleaseQA")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let result: [String: Any] = ["backend": "metal", "thinking_only": firstReply.text.isEmpty,
            "first_input_tokens": firstUsage.inputTokens, "first_generated_tokens": firstUsage.generatedTokens,
            "next_input_tokens": viewModel.contextUsage.inputTokens, "next_generated_tokens": viewModel.contextUsage.generatedTokens,
            "reply": reply, "retained_requests": ConversationContextBuilder.turns(from: viewModel.messages).count]
        try JSONSerialization.data(withJSONObject: result, options: .prettyPrinted)
            .write(to: directory.appendingPathComponent("context-unanswered-retention.json"))
    }

    @Test func metalChatStopsSavesAndRegeneratesWithoutDuplicatingTheQuestion() async throws {
        try await waitUntil(attempts: 500) { UIApplication.shared.applicationState == .active }
        let model = ModelInfo(modelName: "UsabilityQA-" + UUID().uuidString + "-Qwen3.5-0.8B-MNN", tags: ["Think"])
        try prepareUsabilityModel(model, maxNewTokens: 512)
        let config = ModelConfigManager(modelPath: model.localPath)
        config.updateEnableThinking(true)
        config.updateConversationContextEnabled(true)
        config.updateContextTokenLimit(4096)
        let id = UUID().uuidString
        let raw = [HistoryMessage(id: UUID().uuidString, content: "我的暗号是松果73。", images: nil, audio: nil, isUser: true, createdAt: Date()),
                   HistoryMessage(id: UUID().uuidString, content: "我记住了你的暗号。", images: nil, audio: nil, isUser: false, createdAt: Date())]
        let history = ChatHistory(id: id, modelInfo: model, messages: raw, createdAt: Date(), updatedAt: Date())
        let chat = LLMChatViewModel(modelInfo: model, history: history)
        defer { cleanUsabilityModel(model, chat: chat) }
        chat.onStart()
        try await waitUntil(attempts: 5_000) { chat.isModelLoaded || chat.loadFailed }
        #expect(chat.isModelLoaded)
        guard chat.isModelLoaded else { return }
        chat.sendUserPrompt("请深入推理如何证明素数有无穷多个，比较三个证明方法，再只回答我的暗号。")
        // The second tap must be rejected even while the first request is preparing.
        chat.sendUserPrompt("这个问题不应该被发送。")
        try await waitUntil(attempts: 3_000) { !(chat.messages.last?.thinkText ?? "").isEmpty }
        try await capture("chat-stop-zh", view: VStack {
            ThinkView(thinkContent: chat.messages.last?.thinkText ?? "", thinkingFinished: false, messageId: "UsabilityQA")
            Spacer()
            DemoInputBar(isEnabled: false, isProcessing: true, isStopping: false, onStop: {}) { _ in }
        })
        #expect(chat.isProcessing)
        chat.cancelGeneration()
        #expect(chat.isStopping)
        try await waitUntil(attempts: 3_000) { !chat.isProcessing }
        #expect(!chat.isStopping)
        let generated = chat.contextUsage.generatedTokens
        #expect(generated > 0 && generated < 512)
        let stopped = try #require(chat.messages.last(where: { $0.user.id == "2" }))
        #expect(!(stopped.thinkText ?? "").isEmpty)
        #expect(chat.canGenerateDirectAnswer)
        try await waitUntil(attempts: 500) {
            ChatHistoryManager.shared.getAllHistory().first(where: { $0.id == id })?.messages.last(where: { $0.role == "2" })?.thinkText == stopped.thinkText
        }
        let saved = try #require(ChatHistoryManager.shared.getAllHistory().first(where: { $0.id == id }))
        let restored = LLMChatData(modelInfo: model).greatingMessage(historyMessages: saved.messages).map { $0.toChatMessage() }
        #expect(restored.last(where: { $0.user.id == "2" })?.thinkText == stopped.thinkText)
        #expect(ConversationContextBuilder.turns(from: restored).count == 1)
        try await capture("chat-retry-zh", view: VStack {
            ThinkView(thinkContent: stopped.thinkText ?? "", thinkingFinished: true, messageId: "UsabilityQA")
            Spacer()
            ChatResponseActionsView(viewModel: chat)
        })
        chat.regenerateLastResponse(directAnswer: true)
        #expect(chat.isProcessing)
        try await waitUntil(attempts: 5_000) { !chat.isProcessing }
        let answer = try #require(chat.messages.last(where: { $0.user.id == "2" }))
        #expect(!answer.text.isEmpty)
        #expect((answer.thinkText ?? "").isEmpty)
        #expect(chat.isThinkingModeEnabled)
        #expect(config.readEnableThinkingOverride() == true)
        let thinkingPreferencePreserved = chat.isThinkingModeEnabled && config.readEnableThinkingOverride() == true
        #expect(chat.messages.filter { $0.user.isCurrentUser }.count == 2)
        #expect(!chat.messages.contains { $0.text == "这个问题不应该被发送。" })
        #expect(ConversationContextBuilder.turns(from: chat.messages).count == 2)
        chat.toggleThinkingMode()
        chat.sendUserPrompt("我的暗号是什么？只回答暗号。")
        try await waitUntil(attempts: 3_000) { !chat.isProcessing }
        chat.regenerateLastResponse()
        try await waitUntil(attempts: 3_000) { !chat.isProcessing }
        let reply = chat.messages.last(where: { $0.user.id == "2" })?.text ?? ""
        #expect(reply.filter { !$0.isWhitespace }.contains("松果73"))
        #expect(chat.messages.filter { $0.user.isCurrentUser }.count == 3)
        chat.toggleThinkingMode()
        config.updateMaxNewTokens(8)
        chat.setModelConfig()
        chat.sendUserPrompt("我的暗号是什么？先仔细检查历史并推理，再给出答案。")
        try await waitUntil(attempts: 3_000) { !chat.isProcessing }
        #expect(chat.canGenerateDirectAnswer)
        chat.regenerateLastResponse(directAnswer: true)
        try await waitUntil(attempts: 3_000) { !chat.isProcessing }
        let directReply = chat.messages.last(where: { $0.user.id == "2" })?.text ?? ""
        #expect(directReply.filter { !$0.isWhitespace }.contains("松果73"))
        #expect(chat.isThinkingModeEnabled)
        #expect(chat.messages.filter { $0.user.isCurrentUser }.count == 4)
        chat.handleScenePhase(.inactive)
        #expect(!chat.historySaveFailed)
        let final = try #require(ChatHistoryManager.shared.getAllHistory().first(where: { $0.id == id }))
        #expect(final.messages.last(where: { $0.role == "2" })?.content == directReply)
        // A stop before native dispatch must not restart generation. Saving
        // the new conversation must also leave the previous chat intact.
        chat.resetConversation()
        chat.sendUserPrompt("这个新对话会在准备阶段停止。")
        chat.handleScenePhase(.background)
        try await waitUntil { !chat.isProcessing }
        #expect(chat.contextUsage.generatedTokens == 0)
        try await waitUntil(attempts: 500) {
            ChatHistoryManager.shared.getAllHistory().filter { $0.modelId == model.id }.count == 2
        }
        let histories = ChatHistoryManager.shared.getAllHistory().filter { $0.modelId == model.id }
        #expect(histories.first(where: { $0.id == id })?.messages.last(where: { $0.role == "2" })?.content == directReply)
        #expect(histories.first(where: { $0.id != id })?.messages.filter(\.isUser).count == 1)
        let evidence: [String: Any] = ["backend": "metal", "stopped_tokens": generated,
            "saved_thinking_characters": stopped.thinkText?.count ?? 0, "direct_answer_after_stop": answer.text,
            "direct_answer_after_thinking_limit": directReply,
            "thinking_preference_preserved": thinkingPreferencePreserved,
            "regenerated_reply": reply, "user_messages_after_regeneration": 4,
            "saved_before_leaving_chat": true, "background_stop_before_dispatch": true,
            "new_conversation_saved_separately": histories.count == 2]
        try saveUsabilityEvidence("chat-usability.json", evidence)
    }

    @Test func modelLoadFailureCanRetryAfterItsFilesAreRestored() async throws {
        let model = ModelInfo(modelName: "LoadRetryQA-" + UUID().uuidString + "-Qwen3.5-0.8B-MNN", tags: ["Think"])
        try FileManager.default.createDirectory(atPath: model.localPath, withIntermediateDirectories: true)
        ModelConfigManager(modelPath: model.localPath).updateEnableThinking(false)
        let chat = LLMChatViewModel(modelInfo: model)
        defer { cleanUsabilityModel(model, chat: chat) }
        chat.onStart()
        try await waitUntil(attempts: 5_000) { chat.loadFailed }
        #expect(!chat.isModelLoaded)
        #expect(chat.chatInputUnavilable)
        try await capture("chat-load-failed-zh", view: ZStack {
            Color.black.opacity(0.4).ignoresSafeArea()
            ModelLoadFailureView(modelName: model.name, onRetry: {}, onChooseModel: {})
        })
        try prepareUsabilityModel(model, maxNewTokens: 32)
        chat.retryModelLoad()
        #expect(!chat.loadFailed)
        try await waitUntil(attempts: 5_000) { chat.isModelLoaded || chat.loadFailed }
        #expect(chat.isModelLoaded)
        guard chat.isModelLoaded else { return }
        chat.sendUserPrompt("请只回答 OK。")
        try await waitUntil(attempts: 3_000) { !chat.isProcessing }
        let reply = chat.messages.last(where: { $0.user.id == "2" })?.text ?? ""
        #expect(reply == "OK")
        try saveUsabilityEvidence("load-retry.json", ["failure_detected": true, "retry_loaded": chat.isModelLoaded,
            "reply": reply])
    }

    @Test func legacyHistoryAndThinkingMetadataRoundTrip() throws {
        let legacy = try JSONSerialization.data(withJSONObject: ["id": "old", "content": "old reply",
            "isUser": false, "createdAt": 0])
        let decoded = try JSONDecoder().decode(HistoryMessage.self, from: legacy)
        #expect(decoded.thinkText == nil && decoded.role == nil)
        let model = ModelInfo(modelName: "HistoryRoundTripQA-" + UUID().uuidString)
        let id = UUID().uuidString
        let imageURL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("HistoryImageQA-" + UUID().uuidString + ".png")
        let data = UIGraphicsImageRenderer(size: CGSize(width: 4, height: 4)).pngData { context in
            UIColor.red.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 4, height: 4))
        }
        try data.write(to: imageURL)
        defer { try? FileManager.default.removeItem(at: imageURL) }
        let assistant = LLMChatMessage(uid: UUID().uuidString, sender: LLMChatData(modelInfo: model).assistant,
            createdAt: Date(), useMarkdown: false, text: "answer", thinkText: "stored thought", performanceData: "metrics",
            images: [LLMChatImage(id: "image", thumbnail: imageURL, full: imageURL)], videos: [], recording: nil, replyMessage: nil)
            .toChatMessage()
        let raw = [contextMessage("0", "Model loaded"), contextMessage("1", "question"), assistant]
        #expect(ChatHistoryManager.shared.saveChat(historyId: id, modelInfo: model, messages: raw))
        let saved = try #require(ChatHistoryManager.shared.getAllHistory().first(where: { $0.id == id }))
        defer { ChatHistoryManager.shared.deleteHistory(saved) }
        let restored = LLMChatData(modelInfo: model).greatingMessage(historyMessages: saved.messages).map { $0.toChatMessage() }
        #expect(restored.map(\.id) == raw.map(\.id))
        #expect(restored.map { $0.user.id } == ["0", "1", "2"])
        #expect(restored.last?.thinkText == "stored thought")
        #expect(restored.last?.performanceData == raw.last?.performanceData)
        #expect(restored.last?.useMarkdown == false)
        #expect(restored.last?.attachments.count == 1)
        #expect(ConversationContextBuilder.prompts(from: restored) == [["user": "question"], ["assistant": "answer"]])
    }

    @Test func queuedHistoryCheckpointsDoNotRestoreDeletedChats() async throws {
        let model = ModelInfo(modelName: "CheckpointOrderingQA-" + UUID().uuidString)
        let id = UUID().uuidString
        let history = ChatHistory(id: id, modelInfo: model, messages: [], createdAt: Date(), updatedAt: Date())
        let messages = [contextMessage("1", "question"), contextMessage("2", "first answer")]
        var completions = 0
        for _ in 0..<3 {
            ChatHistoryManager.shared.saveChatAsync(historyId: id, modelInfo: model, messages: messages) { success in
                #expect(success)
                completions += 1
            }
        }
        ChatHistoryManager.shared.deleteHistory(history)
        try await waitUntil { completions == 3 }
        #expect(!ChatHistoryManager.shared.getAllHistory().contains { $0.id == id })
    }

    @Test func chatWorkspaceSwitchesModelsWithoutLosingConversationOrDraft() async throws {
        try await waitUntil(attempts: 500) { UIApplication.shared.applicationState == .active }
        let suffix = UUID().uuidString
        let a = ModelInfo(modelName: "WorkspaceQA-A-" + suffix + "-Qwen3.5-0.8B-MNN", tags: ["Think"], isDownloaded: true)
        let b = ModelInfo(modelName: "WorkspaceQA-B-" + suffix + "-Qwen3.5-0.8B-MNN", tags: ["Think"], isDownloaded: true)
        let suite = "WorkspaceQA-" + suffix
        let preferences = try #require(UserDefaults(suiteName: suite))
        var workspace = ChatWorkspaceModel(preferences: preferences)
        defer {
            cleanWorkspaceModels([a, b], workspace: workspace)
            preferences.removePersistentDomain(forName: suite)
        }
        for (model, limit) in [(a, 4096), (b, 2048)] {
            try prepareUsabilityModel(model, maxNewTokens: 64)
            let config = ModelConfigManager(modelPath: model.localPath)
            config.updateEnableThinking(false)
            config.updateConversationContextEnabled(true)
            config.updateContextTokenLimit(limit)
        }
        ModelNameStore.shared.setName("Qwen3.5 · 0.8B", for: a)
        ModelNameStore.shared.setName("Qwen3.5 · 0.8B（备用）", for: b)
        let listing = ModelListViewModel(modelClient: ControlledModelClient(), fetchOnInit: false)
        listing.models = [a, b]
        workspace.updateModels(listing.models)
        let first = try #require(workspace.currentChat)
        try await waitUntil(attempts: 5_000) { first.isModelLoaded || first.loadFailed }
        #expect(first.isModelLoaded)
        guard first.isModelLoaded else { return }
        let historyID = first.currentHistoryIdentifier
        first.sendUserPrompt("我的暗号是松果73。请记住它，只回答记住了。")
        // Reject replacement while the engine is still preparing the request.
        #expect(!workspace.selectModel(b))
        #expect(!workspace.newConversation())
        #expect(workspace.currentChat === first)
        workspace.errorMessage = nil
        try await waitUntil(attempts: 5_000) { !first.isProcessing }
        let initialReply = first.messages.last(where: { $0.user.id == "2" })?.text ?? ""
        #expect(!initialReply.isEmpty)
        workspace.draftText = "请帮我写一个周末计划"
        try await capture("workspace-chat-draft-zh", view: MainTabView(modelList: listing, workspace: workspace)) { window in
            #expect(textInputs(in: window).contains(workspace.draftText))
        }
        #expect(first.isModelLoaded) // Showing and dismissing another view must not stop the engine.
        #expect(workspace.selectModel(b))
        let switched = try #require(workspace.currentChat)
        #expect(switched !== first)
        #expect(!first.isModelLoaded)
        #expect(switched.currentHistoryIdentifier == historyID)
        #expect(switched.messages.filter { $0.user.isCurrentUser }.count == 1)
        #expect(workspace.draftText == "请帮我写一个周末计划")
        try await waitUntil(attempts: 5_000) { switched.isModelLoaded || switched.loadFailed }
        #expect(switched.isModelLoaded)
        guard switched.isModelLoaded else { return }
        #expect(switched.contextLimitTokens == 2048)
        try await capture("workspace-switched-model-zh", view: MainTabView(modelList: listing, workspace: workspace)) { window in
            #expect(textInputs(in: window).contains("请帮我写一个周末计划"))
        }
        switched.sendUserPrompt("我的暗号是什么？只回答暗号。")
        try await waitUntil(attempts: 5_000) { !switched.isProcessing }
        let reply = switched.messages.last(where: { $0.user.id == "2" })?.text ?? ""
        #expect(reply.filter { !$0.isWhitespace }.contains("松果73"))
        #expect(switched.messages.filter { $0.user.isCurrentUser }.count == 2)
        workspace.draftText = ""
        workspace.refreshHistories()
        let saved = try #require(workspace.histories.first { $0.id == historyID })
        #expect(saved.modelInfo.modelName == b.modelName)
        try await capture("workspace-chat-zh", view: MainTabView(modelList: listing, workspace: workspace))
        try await capture("workspace-history-zh", view: ChatHistoryDrawer(histories: [saved], selectedID: saved.id,
            onSelect: { _ in }, onDelete: { _ in }, onNewConversation: {}, onModels: {}, onSettings: {},
            onBenchmark: {}, onClose: {}).frame(maxWidth: 360, maxHeight: .infinity, alignment: .leading))
        #expect(workspace.newConversation())
        let emptyID = try #require(workspace.activeHistoryID)
        #expect(emptyID != historyID)
        #expect(workspace.draftText.isEmpty)
        #expect(workspace.openHistory(saved))
        let restored = try #require(workspace.currentChat)
        try await waitUntil(attempts: 5_000) { restored.isModelLoaded || restored.loadFailed }
        #expect(restored.currentHistoryIdentifier == historyID)
        #expect(restored.messages.last(where: { $0.user.id == "2" })?.text == reply)
        #expect(workspace.deleteHistory(saved))
        #expect(workspace.activeHistoryID != historyID)
        workspace.refreshHistories()
        #expect(!workspace.histories.contains { $0.id == historyID })
        // A catalog source change must not change the stored installed-model choice.
        workspace.shutdown()
        workspace = ChatWorkspaceModel(preferences: preferences)
        let sourceChangedB = ModelInfo(modelName: b.modelName, sources: ["ModelScope": "alternate/" + b.modelName], isDownloaded: true)
        workspace.updateModels([a, sourceChangedB])
        #expect(workspace.currentChat?.modelInfo.modelName == b.modelName)
        let reopened = try #require(workspace.currentChat)
        try await waitUntil(attempts: 5_000) { reopened.isModelLoaded || reopened.loadFailed }
        #expect(reopened.isModelLoaded)
        try saveUsabilityEvidence("chat-workspace-switching.json", [
            "backend": "metal", "models_share_test_weights_only": true, "same_chat_id_after_switch": true,
            "old_engine_released": !first.isModelLoaded, "draft_kept_after_switch": true,
            "remembered_after_switch": reply, "restored_latest_history": true,
            "delete_did_not_resurrect_history": !workspace.histories.contains { $0.id == historyID },
            "selection_restored_after_source_change": workspace.currentChat?.modelInfo.modelName == b.modelName])
    }

    @Test func chatWorkspaceDownloadEnablesChatAndMissingHistoryCanChangeModels() async throws {
        try await waitUntil(attempts: 500) { UIApplication.shared.applicationState == .active }
        let suffix = UUID().uuidString
        let pending = ModelInfo(modelName: "WorkspaceDownloadQA-" + suffix + "-Qwen3.5-0.8B-MNN",
                                tags: ["Think"], file_size: 680_000_000)
        let missing = ModelInfo(modelName: "WorkspaceMissingQA-" + suffix)
        let suite = "WorkspaceDownloadQA-" + suffix
        let preferences = try #require(UserDefaults(suiteName: suite))
        let workspace = ChatWorkspaceModel(preferences: preferences)
        let storage = ModelStorageManager.shared
        let originalDownloads = storage.downloadedModels
        defer {
            cleanWorkspaceModels([pending, missing], workspace: workspace)
            storage.downloadedModels = originalDownloads
            preferences.removePersistentDomain(forName: suite)
        }
        try prepareUsabilityModel(pending, maxNewTokens: 64)
        let config = ModelConfigManager(modelPath: pending.localPath)
        config.updateEnableThinking(false)
        config.updateConversationContextEnabled(true)
        config.updateContextTokenLimit(4096)
        ModelNameStore.shared.setName("Qwen3.5 · 0.8B", for: pending)
        let client = ControlledModelClient()
        let listing = ModelListViewModel(modelClient: client, fetchOnInit: false)
        listing.models = [pending]
        workspace.updateModels(listing.models)
        #expect(workspace.currentChat == nil)
        #expect(workspace.availableModels.isEmpty) // Assets alone are not a completed download.
        try await capture("workspace-first-launch-zh", view: MainTabView(modelList: listing, workspace: workspace))
        let failedDownload = Task { await listing.downloadModel(pending) }
        try await waitUntil { client.downloads.count == 1 }
        client.downloads[0].completion.resume(throwing: ModelScopeError.invalidResponse)
        await failedDownload.value
        // Download failures are shown inline on the row with a retry button.
        #expect(!listing.showError)
        guard case let .failed(message, _) = listing.downloadState(for: pending) else {
            Issue.record("Expected an inline download failure")
            return
        }
        #expect(!message.contains("ModelScopeError"))
        #expect(message.contains(AppText.text("error.modelResponse")))
        #expect(listing.currentlyDownloading == nil)
        try await capture("workspace-download-error-zh", view: ChatModelManagementView(viewModel: listing,
                                activeModelName: nil, initialSection: 1, onClose: {}))
        let download = Task { await listing.downloadModel(pending) }
        try await waitUntil { client.downloads.count == 2 }
        client.downloads[1].progress(0.5)
        try await waitUntil { listing.downloadProgress[pending.id] == 0.5 }
        try await capture("workspace-download-zh", view: ChatModelManagementView(viewModel: listing,
                                activeModelName: nil, initialSection: 1, onClose: {}))
        workspace.draftText = "下载前的未发送问题"
        client.downloads[1].completion.resume()
        await download.value
        #expect(listing.models.first?.isDownloaded == true)
        workspace.updateModels(listing.models)
        let ready = try #require(workspace.currentChat)
        try await waitUntil(attempts: 5_000) { ready.isModelLoaded || ready.loadFailed }
        #expect(ready.isModelLoaded)
        guard ready.isModelLoaded else { return }
        #expect(workspace.draftText == "下载前的未发送问题")
        try await capture("workspace-post-download-draft-zh", view: MainTabView(modelList: listing, workspace: workspace)) { window in
            #expect(textInputs(in: window).contains("下载前的未发送问题"))
        }
        workspace.draftText = ""
        try await capture("workspace-welcome-zh", view: MainTabView(modelList: listing, workspace: workspace))
        do {
            let manager = LanguageManager.shared
            let originalLanguage = manager.currentLanguage
            let systemPreferences = UserDefaults.standard
            let originalAppLanguage = systemPreferences.object(forKey: "AppLanguage")
            let originalAppleLanguages = systemPreferences.object(forKey: "AppleLanguages")
            defer {
                manager.applyLanguage(originalLanguage)
                systemPreferences.set(originalAppLanguage, forKey: "AppLanguage")
                systemPreferences.set(originalAppleLanguages, forKey: "AppleLanguages")
            }
            manager.applyLanguage("en")
            try await capture("workspace-welcome-en", view: MainTabView(modelList: listing, workspace: workspace))
        }
        // Replacing a native input must not silently discard an unsent photo/recording.
        workspace.hasDraftAttachments = true
        workspace.draftText = "带附件的草稿"
        let route = workspace.routeID
        #expect(!workspace.newConversation())
        #expect(!workspace.selectModel(missing))
        #expect(workspace.routeID == route)
        #expect(workspace.draftText == "带附件的草稿")
        workspace.hasDraftAttachments = false
        workspace.errorMessage = nil
        let historyID = UUID().uuidString
        let history = ChatHistory(id: historyID, modelInfo: missing, messages: [
            HistoryMessage(id: UUID().uuidString, content: "我的暗号是夜航42。", images: nil, audio: nil, isUser: true, createdAt: Date()),
            HistoryMessage(id: UUID().uuidString, content: "我记住了。", images: nil, audio: nil, isUser: false, createdAt: Date())],
            createdAt: Date(), updatedAt: Date())
        let fixtureMessages = LLMChatData(modelInfo: missing).greatingMessage(historyMessages: history.messages).map { $0.toChatMessage() }
        #expect(ChatHistoryManager.shared.saveChat(historyId: historyID, modelInfo: missing, messages: fixtureMessages))
        #expect(workspace.openHistory(history))
        let unavailable = try #require(workspace.currentChat)
        try await waitUntil(attempts: 5_000) { unavailable.loadFailed }
        #expect(unavailable.messages.contains { $0.text.contains("夜航42") })
        #expect(workspace.draftText.isEmpty)
        #expect(!workspace.selectModel(missing))
        #expect(workspace.currentChat === unavailable)
        workspace.errorMessage = nil
        try await capture("workspace-missing-model-zh", view: MainTabView(modelList: listing, workspace: workspace))
        let recoveredModel = try #require(workspace.availableModels.first)
        #expect(workspace.selectModel(recoveredModel))
        let recovered = try #require(workspace.currentChat)
        try await waitUntil(attempts: 5_000) { recovered.isModelLoaded || recovered.loadFailed }
        #expect(recovered.isModelLoaded)
        #expect(recovered.currentHistoryIdentifier == historyID)
        guard recovered.isModelLoaded else { return }
        recovered.sendUserPrompt("我的暗号是什么？只回答暗号。")
        try await waitUntil(attempts: 5_000) { !recovered.isProcessing }
        let reply = recovered.messages.last(where: { $0.user.id == "2" })?.text ?? ""
        #expect(reply.filter { !$0.isWhitespace }.contains("夜航42"))
        #expect(recovered.messages.filter { $0.user.isCurrentUser }.count == 2)
        config.updateMaxNewTokens(512)
        recovered.toggleThinkingMode()
        recovered.setModelConfig()
        recovered.sendUserPrompt("请深入推理素数有无穷多个的三种证明方法，详细分析每一个步骤。")
        try await waitUntil(attempts: 3_000) { !(recovered.messages.last?.thinkText ?? "").isEmpty }
        try await capture("workspace-thinking-stop-zh", view: MainTabView(modelList: listing, workspace: workspace))
        #expect(recovered.isProcessing)
        recovered.cancelGeneration()
        try await waitUntil(attempts: 3_000) { !recovered.isProcessing }
        let stoppedTokens = recovered.contextUsage.generatedTokens
        #expect(stoppedTokens > 0 && stoppedTokens < 512)
        try saveUsabilityEvidence("chat-workspace-download.json", ["download_callback_controlled": true,
            "backend_after_download": "metal", "download_retry_after_failure": true, "unavailable_history_readable": true,
            "same_history_id_after_recovery": recovered.currentHistoryIdentifier == historyID,
            "remembered_after_recovery": reply, "draft_attachment_prevents_replacement": true,
            "native_input_stop_ui_rendered": true, "thinking_stopped_tokens": stoppedTokens])
    }

    private func cleanWorkspaceModels(_ models: [ModelInfo], workspace: ChatWorkspaceModel) {
        workspace.shutdown()
        let names = Set(models.map(\.modelName))
        for history in ChatHistoryManager.shared.getAllHistory().filter({ names.contains($0.modelInfo.modelName) }) {
            ChatHistoryManager.shared.deleteHistory(history)
        }
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        for model in models {
            ModelNameStore.shared.setName("", for: model)
            ModelStorageManager.shared.clearCachedSize(for: model.modelName)
            try? FileManager.default.removeItem(at: documents.appendingPathComponent("MNNConfigs/" + model.modelName))
            try? FileManager.default.removeItem(atPath: model.localPath)
        }
    }

    private func textInputs(in view: UIView) -> [String] {
        var texts: [String] = []
        if let textView = view as? UITextView { texts.append(textView.text ?? "") }
        if let textField = view as? UITextField { texts.append(textField.text ?? "") }
        return texts + view.subviews.flatMap { textInputs(in: $0) }
    }

    private func prepareUsabilityModel(_ model: ModelInfo, maxNewTokens: Int) throws {
        let source = URL(fileURLWithPath: ModelInfo(modelId: "taobao-mnn/Qwen3.5-0.8B-MNN").localPath)
        let folder = URL(fileURLWithPath: model.localPath)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for file in try FileManager.default.contentsOfDirectory(at: source, includingPropertiesForKeys: nil) where file.lastPathComponent != "config.json" {
            try FileManager.default.createSymbolicLink(at: folder.appendingPathComponent(file.lastPathComponent), withDestinationURL: file)
        }
        var config = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: source.appendingPathComponent("config.json"))) as? [String: Any])
        config["backend_type"] = "metal"
        config["sampler_type"] = "greedy"
        config["max_new_tokens"] = maxNewTokens
        try JSONSerialization.data(withJSONObject: config).write(to: folder.appendingPathComponent("config.json"))
    }

    private func cleanUsabilityModel(_ model: ModelInfo, chat: LLMChatViewModel) {
        chat.onStop()
        for history in ChatHistoryManager.shared.getAllHistory().filter({ $0.modelId == model.id }) {
            ChatHistoryManager.shared.deleteHistory(history)
        }
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try? FileManager.default.removeItem(at: documents.appendingPathComponent("MNNConfigs/" + model.modelName))
        try? FileManager.default.removeItem(atPath: model.localPath)
    }

    private func saveUsabilityEvidence(_ name: String, _ evidence: [String: Any]) throws {
        let directory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("ReleaseQA")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: evidence, options: .prettyPrinted).write(to: directory.appendingPathComponent(name))
    }

    @Test func releaseScreensRenderOnDevice() async throws {
        let manager = LanguageManager.shared
        let originalLanguage = manager.currentLanguage
        let preferences = UserDefaults.standard
        let originalAppLanguage = preferences.object(forKey: "AppLanguage")
        let originalAppleLanguages = preferences.object(forKey: "AppleLanguages")
        defer {
            manager.applyLanguage(originalLanguage)
            preferences.set(originalAppLanguage, forKey: "AppLanguage")
            preferences.set(originalAppleLanguages, forKey: "AppleLanguages")
        }
        manager.applyLanguage("zh-Hans")
        let viewModel = ModelListViewModel(modelClient: ControlledModelClient(), fetchOnInit: false)
        viewModel.models = [ModelInfo(modelName: "Qwen3.5-0.8B-MNN", file_size: 680_000_000)]
        try await capture("first-model-zh", view: NavigationStack {
            List { FirstModelGuide(viewModel: viewModel, browseModels: {}) }
                .listStyle(.plain).navigationTitle("My Model")
                .navigationBarTitleDisplayMode(.inline)
        })
        try await capture("settings-zh", view: NavigationStack { SettingsView() })
        try await capture("privacy-zh", view: NavigationStack { PrivacyPolicyView() })
        try await capture("support-zh", view: NavigationStack { AppSupportView() })
        let chatModel = ModelInfo(modelName: "ContextSettingsQA-" + UUID().uuidString)
        let chat = LLMChatViewModel(modelInfo: chatModel)
        let customDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MNNConfigs/" + chatModel.modelName)
        defer { try? FileManager.default.removeItem(at: customDirectory) }
        chat.isModelLoaded = true
        chat.setConversationContextEnabled(true)
        try await capture("context-settings-zh", view: ModelSettingsView(showSettings: .constant(true), viewModel: chat))
        try await capture("thinking-panel-zh", view: VStack {
            ThinkView(thinkContent: String(repeating: "分析问题，逐步检查已知条件，再整理可以验证的结论。\n", count: 100),
                      thinkingFinished: false, messageId: "ReleaseQA")
            Spacer()
        }.padding())
        manager.applyLanguage("en")
        try await capture("settings-en", view: NavigationStack { SettingsView() })
    }

    private func capture<Content: View>(_ name: String, view: Content, style: UIUserInterfaceStyle = .light,
                                         verify: ((UIWindow) -> Void)? = nil) async throws {
        try await waitUntil(attempts: 500) { UIApplication.shared.applicationState == .active }
        let scene = try #require(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first)
        let originalKeyWindow = scene.windows.first(where: \.isKeyWindow)
        let window = UIWindow(windowScene: scene)
        let controller = UIHostingController(rootView: view.environment(\.locale, Locale(identifier: LanguageManager.shared.currentLanguage)))
        window.rootViewController = controller
        window.windowLevel = .normal + 1
        window.overrideUserInterfaceStyle = style
        window.makeKeyAndVisible()
        defer {
            window.isHidden = true
            originalKeyWindow?.makeKeyAndVisible()
        }
        try await Task.sleep(for: .milliseconds(650))
        controller.view.layoutIfNeeded()
        verify?(window)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let image = UIGraphicsImageRenderer(bounds: window.bounds, format: format).image { _ in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }
        let data = try #require(image.pngData())
        #expect(data.count > 1_000)
        let directory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("ReleaseQA")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try data.write(to: directory.appendingPathComponent(name + ".png"))
    }

    @Test func manualThinkingFoldSurvivesStreamingAndCompletion() {
        var expansion = ThinkingPanelExpansion(thinkingFinished: false)
        #expect(expansion.isExpanded)
        expansion.toggle()
        for _ in 0..<100 {
            expansion.update(thinkingFinished: false)
            #expect(!expansion.isExpanded)
        }
        expansion.update(thinkingFinished: true)
        #expect(!expansion.isExpanded)
        expansion.toggle()
        expansion.update(thinkingFinished: true)
        #expect(expansion.isExpanded)
        var nextMessage = ThinkingPanelExpansion(thinkingFinished: false)
        #expect(nextMessage.isExpanded)
        nextMessage.update(thinkingFinished: true)
        #expect(!nextMessage.isExpanded)
        #expect(!ThinkingPanelExpansion(thinkingFinished: true).isExpanded)
    }

    private func contextMessage(_ role: String, _ text: String, thinking: String = "") -> Message {
        Message(id: UUID().uuidString,
                user: User(id: role, name: role, avatarURL: nil, isCurrentUser: role == "1"),
                text: text, thinkText: thinking, performanceData: "Prefill: not context")
    }

    @Test func conversationContextOmitsThinkingAndTheCurrentRequest() {
        let messages = [contextMessage("0", "Model loaded"), contextMessage("1", "我的暗号是松果73"),
                        contextMessage("2", "记住了", thinking: "private thinking"),
                        contextMessage("1", "还记得吗"), contextMessage("2", "松果73"),
                        contextMessage("1", "current request"), contextMessage("2", "", thinking: "still thinking")]
        #expect(ConversationContextBuilder.prompts(from: messages) == [
            ["user": "我的暗号是松果73"], ["assistant": "记住了"],
            ["user": "还记得吗"], ["assistant": "松果73"]])
        #expect(ConversationContextBuilder.prompts(from: []) == [])
    }

    @Test func conversationContextRetainsUnansweredEarlierRequests() {
        let unfinished = [contextMessage("1", "我的暗号是松果73"),
                          contextMessage("2", "", thinking: String(repeating: "private thought ", count: 4096))]
        #expect(ConversationContextBuilder.prompts(from: unfinished) == [])
        let current = [contextMessage("1", "我的暗号是什么？"), contextMessage("2", "")]
        let previous = ConversationContextBuilder.turns(from: unfinished + current)
        #expect(previous.count == 1)
        #expect(previous.first?.prompts == [["user": "我的暗号是松果73"]])
        #expect(previous.first?.transcript.contains("private thought") == false)
        // Preserve cancelled and failed requests even if they have no assistant
        // bubble, without turning a UI status message into a model answer.
        let failed = [contextMessage("1", "待处理的问题"), contextMessage("0", "已停止")]
        #expect(ConversationContextBuilder.prompts(from: unfinished + failed + current) == [
            ["user": "我的暗号是松果73"], ["user": "待处理的问题"]])
        let finished = [contextMessage("2", "松果73")]
        #expect(ConversationContextBuilder.prompts(from: unfinished + current + finished) == [
            ["user": "我的暗号是松果73"], ["user": "我的暗号是什么？"], ["assistant": "松果73"]])
    }

    @Test func contextSummariesRejectLegacyTurnOffsetsAndPersistTheNewFormat() throws {
        let historyId = "ContextSummaryQA-" + UUID().uuidString
        let folder = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ChatHistories/" + historyId)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let url = folder.appendingPathComponent("context-summary.json")
        var legacy: [String: Any] = ["modelId": "test", "summary": "old memory", "coveredTurns": 2]
        try JSONSerialization.data(withJSONObject: legacy).write(to: url)
        #expect(ContextSummarySnapshot.load(historyId: historyId, modelId: "test") == nil)
        legacy["formatVersion"] = 1
        try JSONSerialization.data(withJSONObject: legacy).write(to: url)
        #expect(ContextSummarySnapshot.load(historyId: historyId, modelId: "test") == nil)
        let fresh = ContextSummarySnapshot(modelId: "test", summary: "updated memory", coveredTurns: 3)
        fresh.save(historyId: historyId)
        #expect(ContextSummarySnapshot.load(historyId: historyId, modelId: "test") == fresh)
        #expect(ContextSummarySnapshot.load(historyId: historyId, modelId: "another model") == nil)
    }

    @Test func contextUsesTokensInsteadOfAnEightRoundOrCharacterCap() {
        let messages = (0..<12).flatMap { [contextMessage("1", "question\($0)"), contextMessage("2", "answer\($0)")] }
        let prompts = ConversationContextBuilder.prompts(from: messages)
        #expect(prompts.count == 24)
        #expect(prompts.first == ["user": "question0"])
        #expect(prompts.last == ["assistant": "answer11"])
        let longTurn = [contextMessage("1", "long question"), contextMessage("2", String(repeating: "字", count: 12_001))]
        #expect(ConversationContextBuilder.prompts(from: messages + longTurn).count == 26)
        #expect(ContextBudget.promptBudget(limit: 4096, maxOutput: 4096) == 3072)
        #expect(ContextBudget.promptBudget(limit: 1024, maxOutput: 128) == 896)
        #expect(ContextBudget.clamp(32_768, modelMaximum: 2048) == 2048)
        #expect(ContextBudget.clamp(32_768, modelMaximum: 262_144) == 32_768)
        #expect(ContextBudget.clamp(131_072, modelMaximum: 262_144) == ContextBudget.maximumLimit)
    }

    @Test func conversationContextPreferencePersistsAndCannotChangeWhileGenerating() {
        let model = ModelInfo(modelName: "ContextPreferenceQA-" + UUID().uuidString)
        let customDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MNNConfigs/" + model.modelName)
        defer { try? FileManager.default.removeItem(at: customDirectory) }
        let viewModel = LLMChatViewModel(modelInfo: model)
        #expect(!viewModel.isConversationContextEnabled)
        viewModel.setConversationContextEnabled(true)
        #expect(ModelConfigManager(modelPath: model.localPath).readConversationContextEnabled())
        #expect(LLMChatViewModel(modelInfo: model).isConversationContextEnabled)
        viewModel.setContextTokenLimit(8192)
        viewModel.setAutoCompressContext(false)
        let restored = LLMChatViewModel(modelInfo: model)
        #expect(restored.contextLimitTokens == 8192)
        #expect(!restored.autoCompressContext)
        viewModel.isProcessing = true
        viewModel.setConversationContextEnabled(false)
        viewModel.setContextTokenLimit(2048)
        viewModel.setAutoCompressContext(true)
        #expect(viewModel.isConversationContextEnabled)
        #expect(viewModel.contextLimitTokens == 8192)
        #expect(!viewModel.autoCompressContext)
        viewModel.isProcessing = false
        viewModel.setConversationContextEnabled(false)
        #expect(!ModelConfigManager(modelPath: model.localPath).readConversationContextEnabled())
    }

    @Test func modelNominalLimitIsSeparateFromPhoneRecommendation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("ContextMetadataQA-" + UUID().uuidString)
        let folder = root.appendingPathComponent("Qwen3.5-0.8B-MNN")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(ContextBudget.modelLimit(path: folder.path)?.tokens == 262_144)
        #expect(ContextBudget.modelLimit(path: folder.path)?.fromOriginalModel == true)
        try Data("{\"text_config\":{\"max_position_embeddings\":8192}}".utf8)
            .write(to: folder.appendingPathComponent("configuration.json"))
        #expect(ContextBudget.modelLimit(path: folder.path)?.tokens == 8192)
        #expect(ContextBudget.modelLimit(path: folder.path)?.fromOriginalModel == false)
        #expect(ContextBudget.modelLimit(path: root.appendingPathComponent("unknown-model").path) == nil)
    }

    @Test func contextMemoryProfileReadsHybridAndLegacyAttentionDimensions() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("ContextMemoryMetadata-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let file = folder.appendingPathComponent("llm_config.json")
        try Data("{\"model_type\":\"qwen3_5\",\"hidden_size\":1024,\"is_visual\":true}".utf8).write(to: file)
        let qwen = try #require(ContextBudget.memoryProfile(path: folder.path))
        #expect(qwen.kvBytesPerToken(usage: ContextUsage()) == 12_288)
        #expect(qwen.kvLayers == 6)
        var highPrecision = ContextUsage()
        highPrecision.kvKeyBytes = 4
        highPrecision.kvValueBytes = 4
        highPrecision.activationBytes = 4
        #expect(qwen.kvBytesPerToken(usage: highPrecision) == 24_576)
        try Data("{\"hidden_size\":960,\"layer_nums\":32,\"key_value_shape\":[2,1,0,5,64]}".utf8).write(to: file)
        let legacy = try #require(ContextBudget.memoryProfile(path: folder.path))
        #expect(legacy.kvBytesPerToken(usage: ContextUsage()) == 40_960)
        var quantized = ContextUsage()
        quantized.kvKeyBytes = 1
        quantized.kvValueBytes = 1
        quantized.kvScaleBytes = 8
        #expect(legacy.kvBytesPerToken(usage: quantized) == 20_736)
        try Data("{\"text_config\":{\"hidden_size\":1024,\"intermediate_size\":3584,\"num_hidden_layers\":24,\"num_attention_heads\":8,\"num_key_value_heads\":2,\"head_dim\":256,\"full_attention_interval\":4}}".utf8)
            .write(to: folder.appendingPathComponent("configuration.json"))
        #expect(ContextBudget.memoryProfile(path: folder.path)?.kvBytesPerToken(usage: ContextUsage()) == 12_288)
    }

    @Test func contextRecommendationUsesCacheWorkspaceAndModelMaximum() throws {
        let small = ContextBudget.MemoryProfile(kvWidth: 3072, kvLayers: 6, attentionHeads: 8,
            hiddenSize: 1024, intermediateSize: 3584, linearWidth: 2048, hasVision: true)
        var usage = ContextUsage()
        usage.availableMemoryBytes = 2700 * 1024 * 1024
        usage.footprintBytes = 350 * 1024 * 1024
        usage.modelBaselineBytes = usage.footprintBytes
        let recommendation = try #require(ContextBudget.recommendation(profile: small, usage: usage, modelMaximum: 262_144))
        #expect(recommendation.tokens == 16_384)
        #expect(ContextBudget.recommendation(profile: small, usage: usage, modelMaximum: 8192)?.tokens == 8192)
        let large = ContextBudget.MemoryProfile(kvWidth: 32 * 32 * 128, kvLayers: 32, attentionHeads: 32,
            hiddenSize: 4096, intermediateSize: 11_008, linearWidth: 0, hasVision: false)
        let largeRecommendation = try #require(ContextBudget.recommendation(profile: large, usage: usage, modelMaximum: 32_768))
        #expect(largeRecommendation.tokens < recommendation.tokens)
        usage.measuredWorkspaceBytesPerToken = 160_000
        #expect(try #require(ContextBudget.recommendation(profile: small, usage: usage, modelMaximum: 262_144)).tokens < recommendation.tokens)
        usage.availableMemoryBytes = 100 * 1024 * 1024
        #expect(ContextBudget.recommendation(profile: small, usage: usage, modelMaximum: 262_144) == nil)
        #expect(ContextBudget.recommendation(profile: nil, usage: ContextUsage(), modelMaximum: nil) == nil)
    }

    @Test func contextRecommendationReusesResidentCacheAndPrefillWorkspace() throws {
        let profile = ContextBudget.MemoryProfile(kvWidth: 3072, kvLayers: 6, attentionHeads: 8,
            hiddenSize: 1024, intermediateSize: 3584, linearWidth: 2048, hasVision: true)
        var usage = ContextUsage()
        usage.availableMemoryBytes = 1340 * 1024 * 1024
        usage.footprintBytes = 1712 * 1024 * 1024
        usage.modelBaselineBytes = 409 * 1024 * 1024
        usage.workspaceTokens = 16_339
        usage.cachedTokens = 16_350
        #expect(ContextBudget.recommendation(profile: profile, usage: usage, modelMaximum: 262_144)?.tokens == 16_384)
        let retained = ContextBudget.additionalBytes(tokens: 16_384, profile: profile, usage: usage)
        usage.cachedTokens = 32
        #expect(ContextBudget.recommendation(profile: profile, usage: usage, modelMaximum: 262_144)?.tokens == 16_384)
        usage.cachedTokens = 0
        usage.workspaceTokens = 0
        #expect(ContextBudget.additionalBytes(tokens: 16_384, profile: profile, usage: usage) > retained)
        #expect(ContextBudget.additionalBytes(tokens: 16_384, prefillTokens: 512, profile: profile, usage: usage)
            < ContextBudget.additionalBytes(tokens: 16_384, profile: profile, usage: usage))
    }

    @Test func contextMemoryCalibrationIgnoresTransientLoadAndReusedBuffers() throws {
        let model = ModelInfo(modelName: "ContextMemoryCalibration-" + UUID().uuidString)
        let folder = URL(fileURLWithPath: model.localPath)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try Data("{\"model_type\":\"qwen3_5\",\"hidden_size\":1024,\"is_visual\":true}".utf8)
            .write(to: folder.appendingPathComponent("llm_config.json"))
        let chat = LLMChatViewModel(modelInfo: model, showsModelLoadingMessages: false)
        var publications = 0
        let subscription = chat.objectWillChange.sink { publications += 1 }
        defer { subscription.cancel() }
        var data: [String: NSNumber] = ["footprintBytes": NSNumber(value: 350 * 1024 * 1024),
            "availableMemoryBytes": NSNumber(value: 2700 * 1024 * 1024),
            "modelBaselineBytes": NSNumber(value: 900 * 1024 * 1024)]
        chat.applyContextUsage(data)
        #expect(publications == 1)
        data["memorySampleID"] = 1
        data["memorySampleInputTokens"] = 219
        data["memorySampleCacheTokens"] = 225
        data["cachedTokens"] = 225
        data["workspaceTokens"] = 219
        data["memorySamplePeakBytes"] = NSNumber(value: 400 * 1024 * 1024)
        data["footprintBytes"] = data["memorySamplePeakBytes"]
        data["availableMemoryBytes"] = NSNumber(value: 2650 * 1024 * 1024)
        data["modelBaselineBytes"] = data["memorySamplePeakBytes"]
        chat.applyContextUsage(data)
        #expect(try #require(chat.contextUsage.modelBaselineBytes) < 400 * 1024 * 1024)
        data["memorySampleID"] = 2
        data["memorySampleInputTokens"] = 8155
        data["memorySampleCacheTokens"] = 8160
        data["cachedTokens"] = 8160
        data["workspaceTokens"] = 8155
        data["memorySamplePeakBytes"] = NSNumber(value: 1050 * 1024 * 1024)
        data["footprintBytes"] = data["memorySamplePeakBytes"]
        data["availableMemoryBytes"] = NSNumber(value: 2000 * 1024 * 1024)
        chat.applyContextUsage(data)
        #expect(chat.recommendedContextTokens == 16_384)
        #expect(chat.contextUsage.measuredWorkspaceBytesPerToken > 0)
        let baseline = chat.contextUsage.modelBaselineBytes
        data["memorySampleID"] = 3
        data["memorySampleInputTokens"] = 18
        data["memorySampleCacheTokens"] = 24
        data["cachedTokens"] = 24
        data["modelBaselineBytes"] = data["memorySamplePeakBytes"]
        chat.applyContextUsage(data)
        #expect(chat.contextUsage.modelBaselineBytes == baseline)
        #expect(chat.recommendedContextTokens == 16_384)
        #expect(publications == 4)
        let workspace = chat.contextUsage.measuredWorkspaceBytesPerToken
        chat.resetConversation()
        chat.applyContextUsage(data)
        #expect(chat.contextUsage.measuredWorkspaceBytesPerToken == workspace)
        #expect(chat.recommendedContextTokens == 16_384)
    }

    @Test func languageChangesUpdateChatStringsWithoutRelaunch() {
        let manager = LanguageManager.shared
        let originalLanguage = manager.currentLanguage
        let preferences = UserDefaults.standard
        let originalAppLanguage = preferences.object(forKey: "AppLanguage")
        let originalAppleLanguages = preferences.object(forKey: "AppleLanguages")
        defer {
            manager.applyLanguage(originalLanguage)
            preferences.set(originalAppLanguage, forKey: "AppLanguage")
            preferences.set(originalAppleLanguages, forKey: "AppleLanguages")
        }

        manager.applyLanguage("zh-Hans")
        #expect(AppText.text("My Model") == "我的模型")
        #expect(ChatTextLocalization.text("Type a message...") == "输入消息…")
        manager.applyLanguage("en")
        #expect(AppText.text("My Model") == "My Model")
        #expect(ChatTextLocalization.text("Type a message...") == "Type a message...")
        #expect(AppText.text("settings.button.privacy") == "Privacy")
        #expect(preferences.string(forKey: "AppLanguage") == "en")
        manager.applyLanguage("unsupported-language")
        #expect(manager.currentLanguage == "en")
        manager.applyLanguage("zh-Hans")
        #expect(AppText.text("settings.button.privacy") == "隐私说明")
    }

    @Test func duplicateDownloadsAndCancelledCallbacksAreIsolated() async throws {
        let client = ControlledModelClient()
        let viewModel = ModelListViewModel(modelClient: client, fetchOnInit: false)
        let first = ModelInfo(modelName: "DownloadRegression-A")
        let second = ModelInfo(modelName: "DownloadRegression-B")
        let firstTask = Task { await viewModel.downloadModel(first) }
        try await waitUntil { client.downloads.count == 1 }

        await viewModel.downloadModel(second)
        #expect(client.downloads.count == 1)
        #expect(viewModel.currentlyDownloading == first.id)
        await viewModel.pauseDownload()
        #expect(viewModel.currentlyDownloading == nil)
        #expect(viewModel.downloadState(for: first) == .paused(0))

        let secondTask = Task { await viewModel.downloadModel(second) }
        try await waitUntil { client.downloads.count == 2 }
        client.downloads[0].progress(0.9)
        client.downloads[0].completion.resume(throwing: ModelScopeError.downloadCancelled)
        await firstTask.value
        await Task.yield()
        #expect(viewModel.currentlyDownloading == second.id)
        #expect(viewModel.downloadProgress[first.id] == 0)
        client.downloads[1].completion.resume(throwing: ModelScopeError.downloadCancelled)
        await secondTask.value
        #expect(viewModel.currentlyDownloading == nil)
    }

    @Test func downloadedModelsSurviveUnavailableAndChangedCatalogs() async throws {
        let name = "OfflineRegression-" + UUID().uuidString
        let model = ModelInfo(modelId: "taobao-mnn/" + name)
        let folder = URL(fileURLWithPath: model.localPath)
        let storage = ModelStorageManager.shared
        let originalDownloads = storage.downloadedModels
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: folder.appendingPathComponent("config.json"))
        storage.markModelAsDownloaded(name)
        defer {
            storage.downloadedModels = originalDownloads
            storage.clearCachedSize(for: name)
            try? FileManager.default.removeItem(at: folder)
        }

        let client = ControlledModelClient()
        let viewModel = ModelListViewModel(modelClient: client, fetchOnInit: false)
        await viewModel.fetchModels()
        #expect(viewModel.models.contains { $0.modelName == name && $0.isDownloaded })
        #expect(!viewModel.isLoadingModels)
        #expect(!viewModel.showError)
        client.catalogAvailable = true
        await viewModel.fetchModels()
        #expect(viewModel.models.contains { $0.modelName == name && $0.isDownloaded })
    }

    private func waitUntil(attempts: Int = 100, _ condition: () -> Bool) async throws {
        for _ in 0..<attempts {
            if condition() { return }
            try await Task.sleep(for: .milliseconds(20))
        }
        Issue.record("Timed out waiting for the controlled download")
        throw ModelScopeError.invalidResponse
    }
}

@MainActor
private final class MetalProbeState {
    var loaded: Bool?
    var finished = false
    var reply = ""
}

private final class ControlledModelClient: ModelClient {
    struct Download {
        let progress: (Double) -> Void
        let completion: CheckedContinuation<Void, Error>
    }
    @MainActor var downloads: [Download] = []
    @MainActor var catalogAvailable = false

    override func getModelInfo() async throws -> TBDataResponse {
        guard await catalogAvailable else { throw ModelScopeError.invalidResponse }
        return TBDataResponse(tagTranslations: [:], quickFilterTags: [], models: [], metadata: nil)
    }

    override func downloadModel(model: ModelInfo, progress: @escaping (Double) -> Void) async throws {
        try await withCheckedThrowingContinuation { completion in
            Task { @MainActor in
                downloads.append(Download(progress: progress, completion: completion))
            }
        }
    }

    override func cancelDownload(for modelId: String) async {}
}

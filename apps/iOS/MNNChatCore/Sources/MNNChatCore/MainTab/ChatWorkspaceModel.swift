import Combine
import Foundation

/// Own one loaded engine. Changing models checkpoints the conversation before
/// releasing its temporary files, then restores the same chat with the new model.
@MainActor
final class ChatWorkspaceModel: ObservableObject {
    @Published private(set) var currentChat: LLMChatViewModel?
    @Published private(set) var routeID = UUID()
    @Published private(set) var availableModels: [ModelInfo] = []
    @Published private(set) var histories: [ChatHistory] = []
    @Published var draftText = ""
    @Published var hasDraftAttachments = false
    @Published var errorMessage: String?

    private let preferences: UserDefaults
    private let preferenceKey = "mnn.chat.selectedModel"
    private var readinessSubscription: AnyCancellable?

    init(preferences: UserDefaults = .standard) { self.preferences = preferences }

    var isBusy: Bool { currentChat?.isProcessing == true }
    var activeHistoryID: String? { currentChat?.currentHistoryIdentifier }

    nonisolated static func modelKey(_ model: ModelInfo) -> String {
        // Catalog IDs depend on the download source; the installed model does not.
        (model.vendor == "Local" ? "local/" : "downloaded/") + model.modelName
    }

    func updateModels(_ models: [ModelInfo]) {
        var seen = Set<String>()
        availableModels = models.filter {
            ($0.isDownloaded || $0.vendor == "Local") && !ModelUtils.isSanaDiffusionModel($0.modelName)
                && !$0.modelName.localizedCaseInsensitiveContains("stable-diffusion")
                && FileManager.default.fileExists(atPath: $0.localPath + "/config.json")
                && seen.insert(Self.modelKey($0)).inserted
        }
        if let chat = currentChat {
            if chat.loadFailed && availableModels.contains(where: { Self.modelKey($0) == Self.modelKey(chat.modelInfo) }) {
                chat.retryModelLoad()
            }
            return
        }
        let preferred = preferences.string(forKey: preferenceKey)
        let model = availableModels.first { Self.modelKey($0) == preferred }
            ?? availableModels.first { $0.modelName == "Qwen3.5-0.8B-MNN" }
            ?? availableModels.first
        if let model { _ = selectModel(model) }
    }

    @discardableResult
    func selectModel(_ model: ModelInfo) -> Bool {
        guard !isBusy else { errorMessage = AppText.text("workspace.busy"); return false }
        guard !hasDraftAttachments else { errorMessage = AppText.text("workspace.draftAttachments"); return false }
        guard availableModels.contains(where: { Self.modelKey($0) == Self.modelKey(model) }),
              FileManager.default.fileExists(atPath: model.localPath + "/config.json") else {
            errorMessage = AppText.text("workspace.modelMissing")
            return false
        }
        if let chat = currentChat, Self.modelKey(chat.modelInfo) == Self.modelKey(model) {
            if chat.loadFailed { chat.retryModelLoad() }
            return true
        }
        var saved: ChatHistory?
        if let chat = currentChat {
            saved = chat.savedConversation()
            guard !chat.historySaveFailed else { errorMessage = AppText.text("workspace.saveFailed"); return false }
            chat.onStop()
        }
        let history = saved.map {
            ChatHistory(id: $0.id, modelInfo: model, messages: $0.messages, createdAt: $0.createdAt, updatedAt: $0.updatedAt)
        }
        installChat(model: model, history: history)
        return true
    }

    @discardableResult
    func openHistory(_ history: ChatHistory) -> Bool {
        guard !isBusy else { errorMessage = AppText.text("workspace.busy"); return false }
        guard !hasDraftAttachments else { errorMessage = AppText.text("workspace.draftAttachments"); return false }
        if history.id == activeHistoryID { return true }
        // The drawer refreshes `histories` (after saving the open chat) each
        // time it opens, so `history` is current; no full re-read is needed.
        let model = availableModels.first { Self.modelKey($0) == Self.modelKey(history.modelInfo) } ?? history.modelInfo
        if let chat = currentChat {
            chat.saveHistoryImmediately()
            guard !chat.historySaveFailed else { errorMessage = AppText.text("workspace.saveFailed"); return false }
            // Same model: swap the conversation and keep the loaded engine.
            // Reloading it took seconds and blocked the main thread.
            if Self.modelKey(chat.modelInfo) == Self.modelKey(model), !chat.loadFailed {
                chat.loadConversation(history)
                draftText = ""
                routeID = UUID()
                return true
            }
            chat.onStop()
        }
        draftText = ""
        installChat(model: model, history: history)
        return true
    }

    @discardableResult
    func newConversation() -> Bool {
        guard !isBusy else { errorMessage = AppText.text("workspace.busy"); return false }
        guard !hasDraftAttachments else { errorMessage = AppText.text("workspace.draftAttachments"); return false }
        currentChat?.saveHistoryImmediately()
        guard currentChat?.historySaveFailed != true else { errorMessage = AppText.text("workspace.saveFailed"); return false }
        currentChat?.resetConversation()
        draftText = ""
        routeID = UUID()
        refreshHistories()
        return true
    }

    func refreshHistories() {
        currentChat?.saveHistoryImmediately()
        histories = ChatHistoryManager.shared.getAllHistory().sorted { $0.updatedAt > $1.updatedAt }
    }

    @discardableResult
    func deleteHistory(_ history: ChatHistory) -> Bool {
        guard !isBusy else { errorMessage = AppText.text("workspace.busy"); return false }
        guard !hasDraftAttachments else { errorMessage = AppText.text("workspace.draftAttachments"); return false }
        if history.id == activeHistoryID {
            guard newConversation() else { return false }
        }
        ChatHistoryManager.shared.deleteHistory(history)
        histories.removeAll { $0.id == history.id }
        return true
    }

    func shutdown() { currentChat?.onStop() }

    /// Unloads `model` if the chat is using it, so its files can be deleted.
    /// The conversation is saved first; `updateModels` later picks another model.
    @discardableResult
    func releaseModel(_ model: ModelInfo) -> Bool {
        guard let chat = currentChat, chat.modelInfo.modelName == model.modelName else { return true }
        guard !isBusy else { errorMessage = AppText.text("workspace.busy"); return false }
        guard !hasDraftAttachments else { errorMessage = AppText.text("workspace.draftAttachments"); return false }
        chat.saveHistoryImmediately()
        guard !chat.historySaveFailed else { errorMessage = AppText.text("workspace.saveFailed"); return false }
        readinessSubscription?.cancel()
        readinessSubscription = nil
        chat.onStop()
        currentChat = nil
        routeID = UUID()
        preferences.removeObject(forKey: preferenceKey)
        return true
    }

    private func installChat(model: ModelInfo, history: ChatHistory?) {
        let chat = LLMChatViewModel(modelInfo: model, history: history, showsModelLoadingMessages: false)
        // The old chat can publish from deinit while currentChat is being set.
        // Cancel its forwarding first, before that setter holds the Published lock.
        readinessSubscription?.cancel()
        readinessSubscription = nil
        currentChat = chat
        routeID = UUID()
        preferences.set(Self.modelKey(model), forKey: preferenceKey)
        readinessSubscription = Publishers.Merge3(chat.$isProcessing, chat.$isModelLoaded, chat.$loadFailed)
            .sink { [weak self] _ in self?.objectWillChange.send() }
        chat.onStart()
    }
}

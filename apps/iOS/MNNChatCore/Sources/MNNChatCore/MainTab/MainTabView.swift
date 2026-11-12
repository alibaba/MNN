import SwiftUI

/// Keep one chat at the root; downloads, history and settings never push a
/// second chat onto an independent tab's navigation stack.
@MainActor
public struct MainTabView: View {
    @ObservedObject private var languageManager = LanguageManager.shared
    @StateObject private var modelList: ModelListViewModel
    @StateObject private var workspace: ChatWorkspaceModel
    @State private var showsHistory = false
    @State private var panel: Panel?
#if DEBUG
    @State private var didRunChatSmoke = false
#endif
    private enum Panel: String, Identifiable {
        case models, downloads, settings, benchmark
        var id: String { rawValue }
    }

    public init() {
        _modelList = StateObject(wrappedValue: ModelListViewModel())
        _workspace = StateObject(wrappedValue: ChatWorkspaceModel())
    }

    init(modelList: ModelListViewModel, workspace: ChatWorkspaceModel) {
        _modelList = StateObject(wrappedValue: modelList)
        _workspace = StateObject(wrappedValue: workspace)
    }

    private var controls: ChatWorkspaceControls {
        ChatWorkspaceControls(models: workspace.availableModels,
            onSelectModel: { _ = workspace.selectModel($0) }, onManageModels: { panel = .downloads },
            onHistory: openHistory, onNewConversation: { _ = workspace.newConversation() },
            onDraftAttachmentsChanged: { workspace.hasDraftAttachments = $0 })
    }

    public var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                NavigationStack {
                    if let chat = workspace.currentChat {
                        LLMChatView(viewModel: chat, draftText: $workspace.draftText, controls: controls)
                            .id(workspace.routeID)
                    } else { emptyChat }
                }
                .disabled(showsHistory).accessibilityHidden(showsHistory)
                if showsHistory {
                    Color.black.opacity(0.25).ignoresSafeArea()
                        .onTapGesture(perform: closeHistory).transition(.opacity)
                    ChatHistoryDrawer(histories: workspace.histories, selectedID: workspace.activeHistoryID,
                        isBusy: workspace.isBusy,
                        onSelect: { if workspace.openHistory($0) { closeHistory() } },
                        onDelete: { _ = workspace.deleteHistory($0) },
                        onNewConversation: { if workspace.newConversation() { closeHistory() } },
                        onModels: { closeHistory(); panel = .models },
                        onSettings: { closeHistory(); panel = .settings },
                        onBenchmark: { closeHistory(); panel = .benchmark }, onClose: closeHistory)
                        .frame(width: min(geometry.size.width * 0.85, 360))
                        .transition(.move(edge: .leading)).zIndex(1)
                }
            }
        }
        .environment(\.locale, Locale(identifier: languageManager.currentLanguage))
        .sheet(item: $panel) { selected in
            switch selected {
            case .models, .downloads:
                ChatModelManagementView(viewModel: modelList, activeModelName: workspace.currentChat?.modelInfo.modelName,
                                        initialSection: selected == .downloads ? 1 : 0,
                                        allowsSelection: !workspace.isBusy && !workspace.hasDraftAttachments,
                                        selectionError: $workspace.errorMessage, onClose: { panel = nil })
            case .settings:
                NavigationStack {
                    SettingsView().toolbar { ToolbarItem(placement: .topBarTrailing) { Button(AppText.text("workspace.close")) { panel = nil } } }
                }
            case .benchmark:
                NavigationStack {
                    BenchmarkView().navigationTitle(AppText.text("Benchmark"))
                        .toolbar { ToolbarItem(placement: .topBarTrailing) { Button(AppText.text("workspace.close")) { panel = nil } } }
                }
            }
        }
        .alert(AppText.text("Error"), isPresented: Binding(
            get: { panel == nil && (workspace.errorMessage != nil || modelList.showError) },
            set: { if !$0 { workspace.errorMessage = nil; modelList.dismissError() } }
        )) {
            Button(AppText.text("OK")) { workspace.errorMessage = nil; modelList.dismissError() }
        } message: { Text(workspace.errorMessage ?? modelList.errorMessage) }
        .task {
            let workspace = workspace
            modelList.prepareModelDeletion = { workspace.releaseModel($0) }
            workspace.updateModels(modelList.models)
            workspace.refreshHistories()
#if DEBUG
            if AgentQA.enabled { await AgentQA.run(workspace: workspace) }
            else { await runChatSmokeIfRequested() }
#endif
        }
        .onChange(of: modelList.models) { _, models in workspace.updateModels(models) }
        .onChange(of: modelList.selectedModel) { _, model in
            if let model, workspace.selectModel(model) { panel = nil }
            modelList.selectedModel = nil
        }
    }

#if DEBUG
    private func runChatSmokeIfRequested() async {
        guard CommandLine.arguments.contains("--chat-smoke-model"), !didRunChatSmoke else { return }
        didRunChatSmoke = true
        for _ in 0..<300 {
            if let chat = workspace.currentChat, chat.isModelLoaded, !chat.isProcessing { break }
            try? await Task.sleep(for: .milliseconds(500))
        }
        guard let chat = workspace.currentChat, chat.isModelLoaded, !chat.isProcessing else {
            print("[ChatSmoke] model load failed or timed out")
            return
        }
        print("[ChatSmoke] loaded: \(chat.modelInfo.modelName)")
        let oldThinking = chat.isThinkingModeEnabled
        let oldContext = chat.isConversationContextEnabled
        chat.isThinkingModeEnabled = false
        chat.configureThinkingMode()
        chat.setConversationContextEnabled(true)
        defer {
            chat.isThinkingModeEnabled = oldThinking
            chat.configureThinkingMode()
            chat.setConversationContextEnabled(oldContext)
        }
        var turns: [[String: String]] = []
        for prompt in ["请只回答 2+3 的计算结果。", "再把刚才的结果乘以 4，只回答数字。"] {
            print("[ChatSmoke] prompt: \(prompt)")
            let previousIDs = Set(chat.messages.map(\.id))
            chat.sendUserPrompt(prompt)
            for _ in 0..<360 {
                try? await Task.sleep(for: .milliseconds(500))
                if !chat.isProcessing { break }
            }
            guard !chat.isProcessing else {
                print("[ChatSmoke] response timed out")
                return
            }
            let response = chat.messages.filter { !previousIDs.contains($0.id) && !$0.user.isCurrentUser }
                .map(\.text).joined(separator: "\n")
            print("[ChatSmoke] response: \(response)")
            turns.append(["prompt": prompt, "response": response])
            chat.saveHistoryImmediately()
            try? await Task.sleep(for: .milliseconds(500))
        }
        let result: [String: Any] = ["model": chat.modelInfo.modelName, "turns": turns,
            "completed": true, "historySaved": !chat.historySaveFailed]
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        if let data = try? JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: documents.appendingPathComponent("chat-smoke-\(chat.modelInfo.modelName).json"), options: .atomic)
        }
        print("[ChatSmoke] completed: \(chat.modelInfo.modelName)")
    }

#endif

    private var emptyChat: some View {
        VStack(spacing: 0) {
            GeometryReader { proxy in
                // Scrolls on small screens once the download card expands.
                ScrollView {
                    ChatWelcomeView(needsModel: true, modelList: modelList, onPrompt: { _ in },
                                    onDownload: { panel = .downloads })
                        .frame(maxWidth: .infinity, minHeight: proxy.size.height)
                }
            }
            VStack(alignment: .leading, spacing: 12) {
                TextField(AppText.text("chat.input.placeholder"), text: $workspace.draftText, axis: .vertical)
                    .lineLimit(1...4).padding(12)
                    .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
                HStack {
                    ChatModelPicker(models: [], selectedModel: nil, onSelect: { _ in }, onManageModels: { panel = .downloads })
                    Spacer()
                    Image(systemName: "arrow.up.circle.fill").font(.system(size: 32)).foregroundStyle(.tertiary)
                }
            }.padding(16)
        }
        .background(Color(uiColor: .systemBackground))
        .navigationTitle(AppText.text("workspace.chatTitle")).navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                Button(action: openHistory) { Image(systemName: "line.3.horizontal") }
                    .accessibilityLabel(AppText.text("ChatHistroyTitle")).accessibilityIdentifier("chat.openHistory")
            }
            ToolbarItem(placement: .topBarTrailing) {
                Button { _ = workspace.newConversation() } label: { Image(systemName: "square.and.pencil") }
                    .accessibilityLabel(AppText.text("workspace.newChat"))
            }
        }
    }

    private func openHistory() {
        NotificationCenter.default.post(name: .dismissKeyboard, object: nil)
        workspace.refreshHistories()
        withAnimation(.easeInOut(duration: 0.22)) { showsHistory = true }
    }
    private func closeHistory() { withAnimation(.easeInOut(duration: 0.22)) { showsHistory = false } }
}

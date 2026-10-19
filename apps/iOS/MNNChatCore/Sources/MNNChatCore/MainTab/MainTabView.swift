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
        }
        .onChange(of: modelList.models) { _, models in workspace.updateModels(models) }
        .onChange(of: modelList.selectedModel) { _, model in
            if let model, workspace.selectModel(model) { panel = nil }
            modelList.selectedModel = nil
        }
    }

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

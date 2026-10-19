import SwiftUI

struct ChatWorkspaceControls {
    let models: [ModelInfo]
    let onSelectModel: (ModelInfo) -> Void
    let onManageModels: () -> Void
    let onHistory: () -> Void
    let onNewConversation: () -> Void
    var onDraftAttachmentsChanged: ((Bool) -> Void)? = nil
}

struct ChatModelPicker: View {
    let models: [ModelInfo]
    let selectedModel: ModelInfo?
    var isBusy = false
    let onSelect: (ModelInfo) -> Void
    let onManageModels: () -> Void
    @ObservedObject private var nameStore = ModelNameStore.shared

    var body: some View {
        Menu {
            Section(AppText.text("workspace.downloadedModels")) {
                if models.isEmpty { Text(AppText.text("workspace.noModels")) }
                ForEach(models, id: \.localPath) { model in
                    Button { onSelect(model) } label: {
                        if let selectedModel, ChatWorkspaceModel.modelKey(model) == ChatWorkspaceModel.modelKey(selectedModel) {
                            Label(model.name, systemImage: "checkmark")
                        } else { Text(model.name) }
                    }.disabled(isBusy)
                }
            }
            Button(action: onManageModels) {
                Label(AppText.text("workspace.downloadModels"), systemImage: "arrow.down.circle")
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "cube")
                Text(selectedModel?.name ?? AppText.text("Choose a model"))
                    .lineLimit(1).truncationMode(.middle)
                Image(systemName: "chevron.down").font(.caption2.bold())
            }
            .font(.subheadline.weight(.medium))
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(Color(uiColor: .secondarySystemBackground), in: Capsule())
            .foregroundStyle(.primary)
        }
        .accessibilityLabel(AppText.text("Choose a model"))
        .accessibilityValue(selectedModel?.name ?? AppText.text("workspace.noModels"))
        .accessibilityIdentifier("chat.modelPicker")
    }
}

struct ChatWelcomeView: View {
    var needsModel = false
    var isReady = true
    /// When set, the first-run screen offers the suggested model in place.
    var modelList: ModelListViewModel? = nil
    let onPrompt: (String) -> Void
    let onDownload: () -> Void

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "sparkles").font(.system(size: 40, weight: .medium))
                .foregroundStyle(Color.customBlue).accessibilityHidden(true)
            Text(AppText.text("workspace.welcome")).font(.title2.bold())
            Text(AppText.text("workspace.private")).font(.subheadline).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            if needsModel {
                Text(AppText.text("workspace.firstModel")).font(.subheadline).foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                if let modelList {
                    RecommendedModelCard(viewModel: modelList)
                    Button(AppText.text("onboarding.browse"), action: onDownload).buttonStyle(.bordered)
                } else {
                    Button(AppText.text("workspace.downloadModels"), action: onDownload).buttonStyle(.borderedProminent)
                }
            } else {
                ForEach(["workspace.prompt.explain", "workspace.prompt.writing", "workspace.prompt.plan"], id: \.self) { key in
                    Button { onPrompt(AppText.text(key)) } label: {
                        HStack {
                            Text(AppText.text(key)).font(.subheadline)
                            Spacer(minLength: 8)
                            Image(systemName: "arrow.up.left").font(.caption)
                        }
                        .padding(12)
                        .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
                        .foregroundStyle(.primary)
                    }.buttonStyle(.plain).disabled(!isReady)
                }
            }
        }
        .padding(.horizontal, 28).frame(maxWidth: 440)
    }
}

struct ChatHistoryDrawer: View {
    let histories: [ChatHistory]
    let selectedID: String?
    var isBusy = false
    let onSelect: (ChatHistory) -> Void
    let onDelete: (ChatHistory) -> Void
    let onNewConversation: () -> Void
    let onModels: () -> Void
    let onSettings: () -> Void
    let onBenchmark: () -> Void
    let onClose: () -> Void
    @State private var searchText = ""
    @State private var pendingDelete: ChatHistory?

    private var visibleHistories: [ChatHistory] {
        histories.filter { history in
            searchText.isEmpty || history.modelName.localizedCaseInsensitiveContains(searchText)
                || history.messages.contains { $0.content.localizedCaseInsensitiveContains(searchText) }
        }
    }

    /// Swiping left closes the drawer. It is attached to the header and
    /// footer only, so that swiping a conversation row reveals its actions.
    private var closeSwipe: some Gesture {
        DragGesture(minimumDistance: 20).onEnded {
            if $0.translation.width < -70 && abs($0.translation.width) > abs($0.translation.height) { onClose() }
        }
    }

    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Text(AppText.text("workspace.chatTitle")).font(.title2.bold())
                Spacer()
                Button(action: onClose) { Image(systemName: "xmark") }
                    .accessibilityLabel(AppText.text("workspace.close"))
            }
            .contentShape(Rectangle())
            .simultaneousGesture(closeSwipe)
            Button(action: onNewConversation) {
                Label(AppText.text("workspace.newChat"), systemImage: "square.and.pencil")
                    .frame(maxWidth: .infinity, alignment: .leading).padding(12)
                    .background(Color.customBlue.opacity(0.1), in: RoundedRectangle(cornerRadius: 12))
            }.disabled(isBusy).accessibilityIdentifier("history.newChat")
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField(AppText.text("workspace.searchHistory"), text: $searchText).textInputAutocapitalization(.never)
            }.padding(10).background(.background, in: RoundedRectangle(cornerRadius: 10))
            // A List (not a LazyVStack) so rows support swipe actions.
            List {
                if visibleHistories.isEmpty {
                    Text(AppText.text(searchText.isEmpty ? "workspace.emptyHistory" : "workspace.noHistoryResults"))
                        .font(.subheadline).foregroundStyle(.secondary).padding(.vertical, 32)
                        .frame(maxWidth: .infinity)
                        .listRowBackground(Color.clear)
                        .listRowSeparator(.hidden)
                }
                ForEach(visibleHistories) { history in
                    Button { onSelect(history) } label: {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(history.messages.first(where: { $0.isUser && !$0.content.isEmpty })?.content
                                 ?? AppText.text("workspace.newChat"))
                                .font(.subheadline.weight(.medium)).lineLimit(2)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            HStack {
                                Text(history.modelName).lineLimit(1)
                                Spacer(minLength: 4)
                                Text(history.updatedAt.formatAgo())
                            }.font(.caption2).foregroundStyle(.secondary)
                        }
                        .padding(12)
                        .background(history.id == selectedID ? Color.customBlue.opacity(0.12) : .clear,
                                    in: RoundedRectangle(cornerRadius: 12))
                        .foregroundStyle(.primary)
                    }
                    .buttonStyle(.plain).disabled(isBusy)
                    .listRowInsets(EdgeInsets(top: 3, leading: 0, bottom: 3, trailing: 0))
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
                    // Not `role: .destructive`: deletion is confirmed first,
                    // and that role would animate the row away immediately.
                    .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                        Button { pendingDelete = history } label: {
                            Label(AppText.text("Delete"), systemImage: "trash")
                        }
                        .tint(.red)
                        .disabled(isBusy)
                    }
                    .contextMenu {
                        Button(role: .destructive) { pendingDelete = history } label: {
                            Label(AppText.text("Delete History"), systemImage: "trash")
                        }.disabled(isBusy)
                    }
                }
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
            Divider()
            VStack(alignment: .leading, spacing: 20) {
                Button(action: onModels) { Label(AppText.text("workspace.modelManagement"), systemImage: "square.stack.3d.up") }
                HStack {
                    Button(action: onSettings) { Label(AppText.text("Settings"), systemImage: "gearshape") }
                    Spacer()
                    Button(action: onBenchmark) { Image(systemName: "chart.bar") }
                        .accessibilityLabel(AppText.text("Benchmark"))
                }
            }.font(.subheadline).foregroundStyle(.primary)
            .contentShape(Rectangle())
            .simultaneousGesture(closeSwipe)
        }
        .padding(.horizontal, 18).padding(.vertical, 20)
        .background(Color(uiColor: .systemGroupedBackground))
        .confirmationDialog(AppText.text("Delete History"), isPresented: Binding(
            get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }
        ), titleVisibility: .visible) {
            Button(AppText.text("button.delete"), role: .destructive) {
                if let history = pendingDelete { onDelete(history) }
                pendingDelete = nil
            }
            Button(AppText.text("Cancel"), role: .cancel) { pendingDelete = nil }
        } message: { Text(AppText.text("Are you sure you want to delete this history?")) }
        .accessibilityIdentifier("chat.historyDrawer")
    }
}

struct ChatModelManagementView: View {
    @ObservedObject var viewModel: ModelListViewModel
    let activeModelName: String?
    let allowsSelection: Bool
    let onClose: () -> Void
    @Binding private var selectionError: String?
    @State private var section: Int

    init(viewModel: ModelListViewModel, activeModelName: String?, initialSection: Int = 0,
         allowsSelection: Bool = true, selectionError: Binding<String?> = .constant(nil), onClose: @escaping () -> Void) {
        self.viewModel = viewModel
        self.activeModelName = activeModelName
        self.allowsSelection = allowsSelection
        _selectionError = selectionError
        self.onClose = onClose
        _section = State(initialValue: initialSection)
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                Picker(AppText.text("workspace.modelManagement"), selection: $section) {
                    Text(AppText.text("workspace.downloadedModels")).tag(0)
                    Text(AppText.text("workspace.downloadModels")).tag(1)
                }.pickerStyle(.segmented).padding()
                if section == 0 {
                    LocalModelListView(viewModel: viewModel, browseModels: { section = 1 }, activeModelName: activeModelName,
                                       allowsSelection: allowsSelection)
                } else { ModelListView(viewModel: viewModel, activeModelName: activeModelName) }
            }
            .navigationTitle(AppText.text("workspace.modelManagement")).navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .topBarTrailing) { Button(AppText.text("workspace.close"), action: onClose) } }
            .alert(AppText.text("Error"), isPresented: Binding(
                get: { viewModel.showError || selectionError != nil },
                set: { if !$0 { viewModel.dismissError(); selectionError = nil } }
            )) {
                Button(AppText.text("OK")) { viewModel.dismissError(); selectionError = nil }
            } message: { Text(selectionError ?? viewModel.errorMessage) }
        }
    }
}

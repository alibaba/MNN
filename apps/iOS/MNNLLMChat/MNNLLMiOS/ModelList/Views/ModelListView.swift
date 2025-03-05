//
//  ModelListView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import SwiftUI

struct ModelListView: View {
    
    @State private var scrollOffset: CGFloat = 0
    @State private var showHelp = false
    @State private var showUserGuide = false
    @State private var showHistory = false
    @State private var selectedHistory: ChatHistory?
    @State private var histories: [ChatHistory] = []
    @State private var showSettings = false
    @State private var showWebView = false
    @State private var webViewURL: URL?
    
    @StateObject private var viewModel = ModelListViewModel()
    
    var body: some View {
        ZStack {
            NavigationView {
                List {
                    SearchBar(text: $viewModel.searchText)
                        .listRowInsets(EdgeInsets())
                        .listRowSeparator(.hidden)
                        .padding(.horizontal)
                        
                    ForEach(viewModel.filteredModels, id: \.modelId) { model in
                        ModelRowView(model: model,
                                   downloadProgress: viewModel.downloadProgress[model.modelId] ?? 0,
                                   isDownloading: viewModel.currentlyDownloading == model.modelId,
                                   isOtherDownloading: viewModel.currentlyDownloading != nil) {
                            if model.isDownloaded {
                                viewModel.selectModel(model)
                            } else {
                                Task {
                                    await viewModel.downloadModel(model)
                                }
                            }
                        }
                        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                            if model.isDownloaded {
                                Button(role: .destructive) {
                                    Task {
                                        await viewModel.deleteModel(model)
                                    }
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                        }
                    }
                }
                .listStyle(.plain)
                .navigationTitle("Models")
                .navigationBarTitleDisplayMode(.large)
                .navigationBarItems(
                    leading: Button(action: {
                        showHistory.toggle()
                        updateHistory()
                    }) {
                        Image(systemName: "clock.arrow.circlepath")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 22, height: 22)
                    },
                    trailing: settingsButton
                )
                .sheet(isPresented: $showHelp) {
                    HelpView()
                }
                .sheet(isPresented: $showWebView) {
                    if let url = webViewURL {
                        WebView(url: url)
                    }
                }
                .refreshable {
                    await viewModel.fetchModels()
                }
                .alert("Error", isPresented: $viewModel.showError) {
                    Button("OK", role: .cancel) {}
                } message: {
                    Text(viewModel.errorMessage)
                }
                .background(
                    NavigationLink(
                        destination: {
                            if let selectedModel = viewModel.selectedModel {
                                return AnyView(LLMChatView(modelInfo: selectedModel))
                            } else if let selectedHistory = selectedHistory {
                                return AnyView(LLMChatView(modelInfo: ModelInfo(
                                    modelId: selectedHistory.modelId,
                                    createdAt: selectedHistory.createdAt.formatAgo(),
                                    downloads: 0,
                                    tags: [],
                                    isDownloaded: true
                                ), history: selectedHistory))
                            }
                            return AnyView(EmptyView())
                        }(),
                        isActive: Binding(
                            get: { viewModel.selectedModel != nil || selectedHistory != nil },
                            set: { if !$0 { viewModel.selectedModel = nil; selectedHistory = nil } }
                        )
                    ) {
                        EmptyView()
                    }
                )
                .onAppear {
                    checkFirstLaunch()
                }
                .alert(isPresented: $showUserGuide) {
                    Alert(
                        title: Text("User Guide"),
                        message: Text("""
                        This is a local large model application that requires certain performance from your device.
                        It is recommended to choose different model sizes based on your device's memory. 
                        
                        The model recommendations for iPhone are as follows:
                        - For 8GB of RAM, models up to 8B are recommended (e.g., iPhone 16 Pro).
                        - For 6GB of RAM, models up to 3B are recommended (e.g., iPhone 15 Pro).
                        - For 4GB of RAM, models up to 1B or smaller are recommended (e.g., iPhone 13).
                        
                        Choosing a model that is too large may cause insufficient memory and crashes.
                        """),
                        dismissButton: .default(Text("OK"))
                    )
                }
            }
            .disabled(showHistory)
            
            
            if showHistory {
                Color.black.opacity(0.5)
                    .edgesIgnoringSafeArea(.all)
                    .onTapGesture {
                        withAnimation {
                            showHistory = false
                        }
                    }
            }
            
            SideMenuView(isOpen: $showHistory, selectedHistory: $selectedHistory, histories: $histories)
                .edgesIgnoringSafeArea(.all)
        }
        .onAppear {
            updateHistory()
        }
        .actionSheet(isPresented: $showSettings) {
            ActionSheet(title: Text("Settings"), buttons: [
                .default(Text("Report an Issue")) {
                    webViewURL = URL(string: "https://github.com/alibaba/MNN/issues")
                    showWebView = true
                },
                .default(Text("Go to MNN Homepage")) {
                    webViewURL = URL(string: "https://github.com/alibaba/MNN")
                    showWebView = true
                },
                .default(Text(ModelSource.modelScope.description)) {
                    ModelSourceManager.shared.updateSelectedSource(.modelScope)
                },
                .default(Text(ModelSource.modeler.description)) {
                    ModelSourceManager.shared.updateSelectedSource(.modeler)
                },
                .default(Text(ModelSource.huggingFace.description)) {
                    ModelSourceManager.shared.updateSelectedSource(.huggingFace)
                },
                .cancel()
            ])
        }
    }
    
    private func updateHistory() {
        histories = ChatHistoryManager.shared.getAllHistory()
    }
    
    private func checkFirstLaunch() {
        let hasLaunchedBefore = UserDefaults.standard.bool(forKey: "hasLaunchedBefore")
        if !hasLaunchedBefore {
            // Show the user guide alert
            showUserGuide = true
            // Set the flag to true so it doesn't show again
            UserDefaults.standard.set(true, forKey: "hasLaunchedBefore")
        }
    }
    
    private var settingsButton: some View {
        Button(action: {
            showSettings.toggle()
        }) {
            Image(systemName: "gear")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 22, height: 22)
        }
    }
}

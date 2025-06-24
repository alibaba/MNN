//
//  MNNLLMiOSApp.swift
//  MainTabView
//
//  Created by 游薪渝(揽清) on 2025/06/20.
//

import SwiftUI

struct MainTabView: View {
    @State private var showHistory = false
    @State private var selectedHistory: ChatHistory? = nil
    @State private var histories: [ChatHistory] = ChatHistoryManager.shared.getAllHistory()
    @State private var showHistoryButton = true
    @State private var showSettings = false
    @State private var showWebView = false
    @State private var webViewURL: URL?
    @StateObject private var modelListViewModel = ModelListViewModel()
    @State private var selectedTab: Int = 0
    @State private var titles = ["本地模型", "模型市场", "Benchmark"]
    
    var body: some View {
        NavigationView {
            ZStack(alignment: .topLeading) {
                NavigationLink(destination: chatDestination, isActive: chatIsActiveBinding) { EmptyView() }
                
                TabView(selection: $selectedTab) {
                    LocalModelListView(viewModel: modelListViewModel)
                        .tabItem {
                            Image(systemName: "house.fill")
                            Text("本地模型")
                        }
                        .tag(0)
                    ModelListView(viewModel: modelListViewModel)
                        .tabItem {
                            Image(systemName: "cart.fill")
                            Text("模型市场")
                        }
                        .tag(1)
                    BenchmarkView()
                        .tabItem {
                            Image(systemName: "clock.fill")
                            Text("Benchmark")
                        }
                        .tag(2)
                }
                
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
            .navigationTitle(titles[selectedTab])
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarHidden(false)
            .onAppear(perform: setupNavigationBarAppearance)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    if showHistoryButton {
                        Button(action: {
                            showHistory = true
                            showHistoryButton = false
                            histories = ChatHistoryManager.shared.getAllHistory()
                        }) {
                            Image(systemName: "sidebar.left")
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 20, height: 20)
                        }
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        showSettings.toggle()
                    }) {
                        Image(systemName: "gear")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 20, height: 20)
                    }
                }
            }
            .onChange(of: showHistory) { newValue in
                if !newValue {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                        withAnimation {
                            showHistoryButton = true
                        }
                    }
                }
            }
            .sheet(isPresented: $showWebView) {
                if let url = webViewURL {
                    WebView(url: url)
                }
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
    }

    @ViewBuilder
    private var chatDestination: some View {
        if let model = modelListViewModel.selectedModel {
            LLMChatView(modelInfo: model)
        } else if let history = selectedHistory {
            let modelInfo = ModelInfo(
                modelId: history.modelId,
                createdAt: "",
                downloads: 0,
                tags: [],
                isDownloaded: true
            )
            LLMChatView(modelInfo: modelInfo, history: history)
        } else {
            EmptyView()
        }
    }
    
    private var chatIsActiveBinding: Binding<Bool> {
        Binding<Bool>(
            get: { modelListViewModel.selectedModel != nil || selectedHistory != nil },
            set: { isActive in
                if !isActive {
                    modelListViewModel.selectedModel = nil
                    selectedHistory = nil
                }
            }
        )
    }
    
    private func setupNavigationBarAppearance() {
        let appearance = UINavigationBarAppearance()
        appearance.configureWithOpaqueBackground()
        appearance.backgroundColor = .white
        appearance.shadowColor = .clear // Optional: remove the shadow
        
        UINavigationBar.appearance().standardAppearance = appearance
        UINavigationBar.appearance().compactAppearance = appearance
        UINavigationBar.appearance().scrollEdgeAppearance = appearance
    }
} 

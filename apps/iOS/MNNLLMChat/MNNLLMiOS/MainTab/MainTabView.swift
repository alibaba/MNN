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
    @State private var navigateToSettings = false  // 新增状态变量
    @StateObject private var modelListViewModel = ModelListViewModel()
    @State private var selectedTab: Int = 0
    @State private var titles = ["本地模型", "模型市场", "TB模型", "Benchmark"]
    
    var body: some View {
        ZStack {
            NavigationView {
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
                    TBModelListView()
                        .tabItem {
                            Image(systemName: "doc.text.fill")
                            Text("TB模型")
                        }
                        .tag(2)
                    BenchmarkView()
                        .tabItem {
                            Image(systemName: "clock.fill")
                            Text("Benchmark")
                        }
                        .tag(3)
                }
                .background(
                    ZStack {
                        NavigationLink(destination: chatDestination, isActive: chatIsActiveBinding) { EmptyView() }
                        NavigationLink(destination: SettingsView(), isActive: $navigateToSettings) { EmptyView() }
                    }
                )
                .navigationTitle(titles[selectedTab])
                .navigationBarTitleDisplayMode(.inline)
                .navigationBarHidden(false)
                .onAppear {
                    setupNavigationBarAppearance()
                    setupTabBarAppearance()
                }
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
                                    .foregroundColor(.black)
                            }
                        }
                    }
                    
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button(action: {
                            if let url = URL(string: "https://github.com/alibaba/MNN") {
                                UIApplication.shared.open(url)
                            }
                        }) {
                            Image(systemName: "star")
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 20, height: 20)
                                .foregroundColor(.black)
                        }
                    }
                }
            }
            .tint(.black)
            
            if showHistory {
                Color.black.opacity(0.5)
                    .edgesIgnoringSafeArea(.all)
                    .onTapGesture {
                        withAnimation {
                            showHistory = false
                        }
                    }
            }
            
            SideMenuView(isOpen: $showHistory, 
                        selectedHistory: $selectedHistory, 
                        histories: $histories,
                        navigateToMainSettings: $navigateToSettings)
                        .edgesIgnoringSafeArea(.all)
        }
        .onChange(of: showHistory) { oldValue, newValue in
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
                    if let model = modelListViewModel.selectedModel {
                        modelListViewModel.recordModelUsage(modelId: model.modelId)
                    }
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
        appearance.shadowColor = .clear
        
        UINavigationBar.appearance().standardAppearance = appearance
        UINavigationBar.appearance().compactAppearance = appearance
        UINavigationBar.appearance().scrollEdgeAppearance = appearance
    }
    
    private func setupTabBarAppearance() {
        let appearance = UITabBarAppearance()
        appearance.configureWithOpaqueBackground()
        
        let selectedColor = UIColor(Color.primaryPurple)
        
        appearance.stackedLayoutAppearance.selected.iconColor = selectedColor
        appearance.stackedLayoutAppearance.selected.titleTextAttributes = [.foregroundColor: selectedColor]

        UITabBar.appearance().standardAppearance = appearance
        UITabBar.appearance().scrollEdgeAppearance = appearance
    }
}
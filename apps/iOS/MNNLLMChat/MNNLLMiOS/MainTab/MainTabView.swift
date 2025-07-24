//
//  MNNLLMiOSApp.swift
//  MainTabView
//
//  Created by 游薪渝(揽清) on 2025/06/20.
//

import SwiftUI

// MainTabView is the primary view of the app, containing the tab bar and navigation for main sections.
struct MainTabView: View {
    // MARK: - State Properties
    
    @State private var showHistory = false
    @State private var selectedHistory: ChatHistory? = nil
    @State private var histories: [ChatHistory] = ChatHistoryManager.shared.getAllHistory()
    @State private var showHistoryButton = true
    @State private var showSettings = false
    @State private var showWebView = false
    @State private var webViewURL: URL?
    @State private var navigateToSettings = false
    @StateObject private var modelListViewModel = ModelListViewModel()
    @State private var selectedTab: Int = 0
    
    private var titles: [String] {
        [
            NSLocalizedString("Local Model", comment: "本地模型标签"),
            NSLocalizedString("Model Market", comment: "模型市场标签"),
            NSLocalizedString("Benchmark", comment: "基准测试标签")
        ]
    }
    
    // MARK: - Body
    
    var body: some View {
        ZStack {
            // Main TabView for navigation between Local Model, Model Market, and Benchmark
            TabView(selection: $selectedTab) {
                NavigationView {
                    LocalModelListView(viewModel: modelListViewModel)
                        .navigationTitle(titles[0])
                        .navigationBarTitleDisplayMode(.inline)
                        .navigationBarHidden(false)
                        .onAppear {
                            setupNavigationBarAppearance()
                        }
                        .toolbar {
                            CommonToolbarView(
                                showHistory: $showHistory,
                                showHistoryButton: $showHistoryButton,
                            )
                        }
                        .background(
                            ZStack {
                                NavigationLink(destination: chatDestination, isActive: chatIsActiveBinding) { EmptyView() }
                                NavigationLink(destination: SettingsView(), isActive: $navigateToSettings) { EmptyView() }
                            }
                        )
                        // Hide TabBar when entering chat or settings view
                        .toolbar((chatIsActiveBinding.wrappedValue || navigateToSettings) ? .hidden : .visible, for: .tabBar)
                }
                .tabItem {
                    Image(systemName: "house.fill")
                    Text(titles[0])
                }
                .tag(0)
                
                NavigationView {
                    ModelListView(viewModel: modelListViewModel)
                        .navigationTitle(titles[1])
                        .navigationBarTitleDisplayMode(.inline)
                        .navigationBarHidden(false)
                        .onAppear {
                            setupNavigationBarAppearance()
                        }
                        .toolbar {
                            CommonToolbarView(
                                showHistory: $showHistory,
                                showHistoryButton: $showHistoryButton,
                            )
                        }
                        .background(
                            ZStack {
                                NavigationLink(destination: chatDestination, isActive: chatIsActiveBinding) { EmptyView() }
                                NavigationLink(destination: SettingsView(), isActive: $navigateToSettings) { EmptyView() }
                            }
                        )
                }
                .tabItem {
                    Image(systemName: "doc.text.fill")
                    Text(titles[1])
                }
                .tag(1)
                
                NavigationView {
                    BenchmarkView()
                        .navigationTitle(titles[2])
                        .navigationBarTitleDisplayMode(.inline)
                        .navigationBarHidden(false)
                        .onAppear {
                            setupNavigationBarAppearance()
                        }
                        .toolbar {
                            CommonToolbarView(
                                showHistory: $showHistory,
                                showHistoryButton: $showHistoryButton,
                            )
                        }
                        .background(
                            ZStack {
                                NavigationLink(destination: chatDestination, isActive: chatIsActiveBinding) { EmptyView() }
                                NavigationLink(destination: SettingsView(), isActive: $navigateToSettings) { EmptyView() }
                            }
                        )
                }
                .tabItem {
                    Image(systemName: "clock.fill")
                    Text(titles[2])
                }
                .tag(2)
            }
            .onAppear {
                setupTabBarAppearance()
            }
            .tint(.black)
            
            // Overlay for dimming the background when history is shown
            if showHistory {
                Color.black.opacity(0.5)
                .edgesIgnoringSafeArea(.all)
                .onTapGesture {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        showHistory = false
                    }
                }
            }
            
            // Side menu for displaying chat history
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

    // MARK: - View Builders
    
    /// Destination view for chat, either from a new model or a history item.
    @ViewBuilder
    private var chatDestination: some View {
        if let model = modelListViewModel.selectedModel {
            LLMChatView(modelInfo: model)
                .navigationBarHidden(false)
                .navigationBarTitleDisplayMode(.inline)
        } else if let history = selectedHistory {
            let modelInfo = ModelInfo(modelId: history.modelId, isDownloaded: true)
            LLMChatView(modelInfo: modelInfo, history: history)
                .navigationBarHidden(false)
                .navigationBarTitleDisplayMode(.inline)
        } else {
            EmptyView()
        }
    }
    
    // MARK: - Bindings
    
    /// Binding to control the activation of the chat view.
    private var chatIsActiveBinding: Binding<Bool> {
        Binding<Bool>(
            get: { 
                return modelListViewModel.selectedModel != nil || selectedHistory != nil
            },
            set: { isActive in
                if !isActive {
                    // Record usage when returning from chat
                    if let model = modelListViewModel.selectedModel {
                        modelListViewModel.recordModelUsage(modelName: model.modelName)
                    }
                    
                    // Clear selections
                    modelListViewModel.selectedModel = nil
                    selectedHistory = nil
                }
            }
        )
    }
    
    // MARK: - Private Methods
    
    /// Configures the appearance of the navigation bar.
    private func setupNavigationBarAppearance() {
        let appearance = UINavigationBarAppearance()
        appearance.configureWithOpaqueBackground()
        appearance.backgroundColor = .white
        appearance.shadowColor = .clear
        
        UINavigationBar.appearance().standardAppearance = appearance
        UINavigationBar.appearance().compactAppearance = appearance
        UINavigationBar.appearance().scrollEdgeAppearance = appearance
    }
    
    /// Configures the appearance of the tab bar.
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

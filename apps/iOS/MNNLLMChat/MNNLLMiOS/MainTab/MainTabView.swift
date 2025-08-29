//
//  MainTabView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/06/20.
//

import SwiftUI

struct MainTabView: View {
    
    @State private var histories: [ChatHistory] = []
    @State private var showHistory = false
    @State private var showHistoryButton = true
    @State private var selectedHistory: ChatHistory? = nil
    
    @State private var showSettings = false
    
    @State private var navigateToSettings = false
    @State private var navigateToChat = false
    
    @State private var selectedTab: Int = 0
    @State private var hasConfiguredAppearance = false
    
    @StateObject private var modelListViewModel = ModelListViewModel()
    
    
    private var titles: [String] {
        [
            NSLocalizedString("Local Model", comment: "本地模型标签"),
            NSLocalizedString("Model Market", comment: "模型市场标签"),
            NSLocalizedString("Benchmark", comment: "基准测试标签")
        ]
    }
    
    var body: some View {
        ZStack {
            
            TabView(selection: $selectedTab) {
                createTabContent(
                    content: LocalModelListView(viewModel: modelListViewModel),
                    title: titles[0],
                    icon: "house.fill",
                    tag: 0
                )
                
                createTabContent(
                    content: ModelListView(viewModel: modelListViewModel),
                    title: titles[1],
                    icon: "doc.text.fill",
                    tag: 1
                )
                
                createTabContent(
                    content: BenchmarkView(),
                    title: titles[2],
                    icon: "clock.fill",
                    tag: 2
                )
            }
            .onAppear {
                setupAppearanceOnce()
                loadHistoriesIfNeeded()
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
            handleHistoryToggle(newValue)
        }
        .onChange(of: modelListViewModel.selectedModel) { oldValue, newValue in
            if newValue != nil {
                navigateToChat = true
            }
        }
        .onChange(of: selectedHistory) { oldValue, newValue in
            if newValue != nil {
                navigateToChat = true
            }
        }
        .onChange(of: navigateToChat) { oldValue, newValue in
            if !newValue && oldValue {
                refreshHistories()
                modelListViewModel.selectedModel = nil
                selectedHistory = nil
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
            LLMChatView(modelInfo: history.modelInfo, history: history)
                .navigationBarHidden(false)
                .navigationBarTitleDisplayMode(.inline)
        } else {
            EmptyView()
        }
    }

    // MARK: - Private Methods
    
    /// Creates a reusable tab content with navigation and common configurations.
    @ViewBuilder
    private func createTabContent<Content: View>(
        content: Content,
        title: String,
        icon: String,
        tag: Int
    ) -> some View {
        NavigationStack {
            content
                .navigationTitle(title)
                .navigationBarTitleDisplayMode(.inline)
                .navigationBarHidden(false)
                .toolbar {
                    CommonToolbarView(
                        showHistory: $showHistory,
                        showHistoryButton: $showHistoryButton
                    )
                }
                .navigationDestination(isPresented: $navigateToChat) {
                    chatDestination
                }
                .navigationDestination(isPresented: $navigateToSettings) {
                    SettingsView()
                }
                .toolbar((navigateToChat || navigateToSettings) ? .hidden : .visible, for: .tabBar)
        }
        .tabItem {
            Image(systemName: icon)
            Text(title)
        }
        .tag(tag)
    }
    
    /// Configures UI appearance only once to prevent memory issues.
    private func setupAppearanceOnce() {
        guard !hasConfiguredAppearance else { return }
        hasConfiguredAppearance = true
        
        setupNavigationBarAppearance()
        setupTabBarAppearance()
    }
    
    /// Loads chat histories if not already loaded.
    private func loadHistoriesIfNeeded() {
        if histories.isEmpty {
            histories = ChatHistoryManager.shared.getAllHistory()
        }
    }
    
    /// Refreshes the histories array.
    private func refreshHistories() {
        histories = ChatHistoryManager.shared.getAllHistory()
    }
    
    /// Handles history toggle with proper memory management.
    private func handleHistoryToggle(_ isShowing: Bool) {
        if isShowing {
            refreshHistories()
        } else {
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 300_000_000) // 0.3 seconds
                withAnimation {
                    self.showHistoryButton = true
                }
            }
        }
    }
    
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

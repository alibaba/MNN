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
    
    var body: some View {
        ZStack(alignment: .topLeading) {
            TabView {
                LocalModelListView()
                    .tabItem {
                        Image(systemName: "house.fill")
                        Text("本地模型")
                    }
                ModelListView()
                    .tabItem {
                        Image(systemName: "cart.fill")
                        Text("模型市场")
                    }
                BenchmarkView()
                    .tabItem {
                        Image(systemName: "clock.fill")
                        Text("Benchmark")
                    }
            }
            
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
                        .padding(EdgeInsets(top: 12, leading: 18, bottom: 0, trailing: 0))
                }
                .zIndex(2)
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
        .onChange(of: showHistory) { newValue in
            if !newValue {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    withAnimation {
                        showHistoryButton = true
                    }
                }
            }
        }
    }
} 

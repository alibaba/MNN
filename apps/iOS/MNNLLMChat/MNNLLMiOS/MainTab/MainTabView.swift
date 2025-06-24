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
            
            HStack {
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
                            .padding(EdgeInsets(top: 12, leading: 0, bottom: 0, trailing: 0))
                    }
                }
                Spacer()
                Button(action: {
                    showSettings.toggle()
                }) {
                    Image(systemName: "gear")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 20, height: 20)
                        .padding(EdgeInsets(top: 12, leading: 0, bottom: 0, trailing: 0))
                }
            }
            .padding(.horizontal, 18)
            .zIndex(2)
            
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

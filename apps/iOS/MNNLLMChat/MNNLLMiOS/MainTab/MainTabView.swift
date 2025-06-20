//
//  MNNLLMiOSApp.swift
//  MainTabView
//
//  Created by 游薪渝(揽清) on 2025/06/20.
//

import SwiftUI

struct MainTabView: View {
    var body: some View {
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
    }
} 

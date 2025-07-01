//
//  SettingsView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/1.
//

import SwiftUI

struct SettingsView: View {
    
    private var sourceManager = ModelSourceManager.shared
    @State private var selectedLanguage = LanguageManager.shared.currentLanguage
    @State private var selectedSource = ModelSourceManager.shared.selectedSource
    @State private var showLanguageAlert = false
    
    private let languages = ["简体中文", "English"]
    
    var body: some View {
        List {
            Section(header: Text("应用")) {
                
                Picker("下载源", selection: $selectedSource) {
                    ForEach(ModelSource.allCases, id: \.self) { source in
                        Text(source.rawValue).tag(source)
                    }
                }
                .onChange(of: selectedSource) { _, newValue in
                    sourceManager.updateSelectedSource(newValue)
                }
                
                Picker("语言", selection: $selectedLanguage) {
                    ForEach(languages, id: \.self) { language in
                        Text(language).tag(language)
                    }
                }
                .onChange(of: selectedLanguage) { _, newValue in
                    if newValue != LanguageManager.shared.currentLanguage {
                        showLanguageAlert = true
                    }
                }
            }
            
            Section(header: Text("关于")) {
                Button(action: {
                    if let url = URL(string: "https://github.com/alibaba/MNN") {
                        UIApplication.shared.open(url)
                    }
                }) {
                    HStack {
                        Text("关于 MNN")
                        Spacer()
                        Image(systemName: "chevron.right")
                            .foregroundColor(.gray)
                            .font(.system(size: 14))
                    }
                    .foregroundColor(.primary)
                }
                
                Button(action: {
                    if let url = URL(string: "https://github.com/alibaba/MNN") {
                        UIApplication.shared.open(url)
                    }
                }) {
                    HStack {
                        Text("反馈问题")
                        Spacer()
                        Image(systemName: "chevron.right")
                            .foregroundColor(.gray)
                            .font(.system(size: 14))
                    }
                    .foregroundColor(.primary)
                }
            }
        }
        .listStyle(InsetGroupedListStyle())
        .navigationTitle("设置")
        .navigationBarTitleDisplayMode(.inline)
        .alert("切换语言", isPresented: $showLanguageAlert) {
            Button("确定") {
                LanguageManager.shared.applyLanguage(selectedLanguage)
                // 重启应用以应用语言更改
                exit(0)
            }
            Button("取消", role: .cancel) {
                // 恢复原来的选择
                selectedLanguage = LanguageManager.shared.currentLanguage
            }
        } message: {
            Text("切换语言需要重启应用，是否继续？")
        }
    }
}

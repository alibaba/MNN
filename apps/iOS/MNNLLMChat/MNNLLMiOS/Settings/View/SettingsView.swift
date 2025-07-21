//
//  SettingsView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/1.
//

import SwiftUI

struct SettingsView: View {
    
    private var sourceManager = ModelSourceManager.shared
    @State private var selectedLanguage = ""
    @State private var selectedSource = ModelSourceManager.shared.selectedSource
    @State private var showLanguageAlert = false
    
    private let languageOptions = LanguageManager.shared.languageOptions
    
    var body: some View {
        List {
            Section(header: Text("settings.section.application")) {
                
                Picker("settings.picker.downloadSource", selection: $selectedSource) {
                    ForEach(ModelSource.allCases, id: \.self) { source in
                        Text(source.rawValue).tag(source)
                    }
                }
                .onChange(of: selectedSource) { _, newValue in
                    sourceManager.updateSelectedSource(newValue)
                }
                
                Picker("settings.picker.language", selection: $selectedLanguage) {
                    ForEach(languageOptions.keys.sorted(), id: \.self) { key in
                        Text(languageOptions[key] ?? "").tag(key)
                    }
                }
                .onChange(of: selectedLanguage) { _, newValue in
                    if newValue != LanguageManager.shared.currentLanguage {
                        showLanguageAlert = true
                    }
                }
            }
            
            Section(header: Text("settings.section.about")) {
                Button(action: {
                    if let url = URL(string: "https://github.com/alibaba/MNN") {
                        UIApplication.shared.open(url)
                    }
                }) {
                    HStack {
                        Text("settings.button.aboutMNN")
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
                        Text("settings.button.reportIssue")
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
        .navigationTitle("settings.navigation.title")
        .navigationBarTitleDisplayMode(.inline)
        .alert("settings.alert.switchLanguage.title", isPresented: $showLanguageAlert) {
            Button("settings.alert.switchLanguage.confirm") {
                LanguageManager.shared.applyLanguage(selectedLanguage)
                // 重启应用以应用语言更改
                exit(0)
            }
            Button("settings.alert.switchLanguage.cancel", role: .cancel) {
                // 恢复原来的选择
                selectedLanguage = LanguageManager.shared.currentLanguage
            }
        } message: {
            Text("settings.alert.switchLanguage.message")
        }
        .onAppear {
            selectedLanguage = LanguageManager.shared.currentLanguage
        }
    }
}

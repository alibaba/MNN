//
//  SettingsView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/1.
//

import SwiftUI

struct SettingsView: View {
    private var sourceManager = ModelSourceManager.shared
    @State private var selectedLanguage = "简体中文"
    @State private var selectedSource = ModelSourceManager.shared.selectedSource
    
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
        
    }
}

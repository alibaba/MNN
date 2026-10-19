//
//  SettingsView.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/1.
//

import Foundation
import SwiftUI

struct SettingsView: View {
    @ObservedObject private var languageManager = LanguageManager.shared
    private var sourceManager = ModelSourceManager.shared
    @State private var selectedLanguage = ""
    @State private var selectedSource = ModelSourceManager.shared.selectedSource
    @State private var showLanguageAlert = false

    private var languageOptions: [String: String] { languageManager.languageOptions }

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

            Section(header: Text(AppText.text("agent.mode"))) {
                NavigationLink(AppText.text("agent.memory.title")) { AgentMemoryView() }
            }

            Section(header: Text("settings.section.about")) {
                NavigationLink("settings.button.privacy") { PrivacyPolicyView() }
                NavigationLink("settings.button.licenses") { OpenSourceLicensesView() }
                NavigationLink("settings.button.help") { AppSupportView() }

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

                HStack {
                    Text("settings.version")
                    Spacer()
                    Text(AppSupportInfo.version).foregroundStyle(.secondary)
                }
            }
        }
        .listStyle(InsetGroupedListStyle())
        .navigationTitle("settings.navigation.title")
        .navigationBarTitleDisplayMode(.inline)
        .alert("settings.alert.switchLanguage.title", isPresented: $showLanguageAlert) {
            Button("settings.alert.switchLanguage.confirm") {
                LanguageManager.shared.applyLanguage(selectedLanguage)
            }
            Button("settings.alert.switchLanguage.cancel", role: .cancel) {
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

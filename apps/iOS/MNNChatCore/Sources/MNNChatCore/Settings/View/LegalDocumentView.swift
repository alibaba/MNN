import SwiftUI

enum LegalDocumentKind: String {
    case privacy, terms, disclaimer, services

    var title: String { AppText.text("legal.\(rawValue).title") }
}

struct LegalDocument: Decodable {
    struct Part: Decodable {
        let title: String
        let paragraphs: [String]
    }

    let sections: [Part]

    static func load(_ kind: LegalDocumentKind, language: String) -> LegalDocument? {
        let language = language.hasPrefix("zh") ? "zh-Hans" : "en"
        guard let url = Bundle.module.url(forResource: "LegalDocuments-\(language)", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let documents = try? JSONDecoder().decode([String: LegalDocument].self, from: data) else { return nil }
        return documents[kind.rawValue]
    }
}

/// Bundled documents remain readable without a network connection.
struct LegalDocumentView: View {
    let kind: LegalDocumentKind
    @ObservedObject private var languageManager = LanguageManager.shared

    var body: some View {
        List {
            if let document = LegalDocument.load(kind, language: languageManager.currentLanguage) {
                Section {
                    if let email = AppSupportInfo.email {
                        Text(email).textSelection(.enabled)
                    }
                }
                ForEach(document.sections.indices, id: \.self) { index in
                    let section = document.sections[index]
                    Section {
                        ForEach(section.paragraphs.indices, id: \.self) { paragraph in
                            Text((try? AttributedString(markdown: section.paragraphs[paragraph],
                                options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
                                ?? AttributedString(section.paragraphs[paragraph]))
                                .textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    } header: { Text(section.title) }
                }
                if kind == .privacy {
                    Section {
                        NavigationLink(LegalDocumentKind.services.title) { LegalDocumentView(kind: .services) }
                        if let url = AppSupportInfo.website("PrivacyPolicyURL") {
                            Link("privacy.online", destination: url)
                        }
                    }
                }
            } else {
                Section {
                    Text(AppText.text("legal.unavailable"))
                    if let email = AppSupportInfo.email { Text(email).textSelection(.enabled) }
                }
            }
        }
        .navigationTitle(kind.title)
        .navigationBarTitleDisplayMode(.inline)
        .accessibilityIdentifier("legal.\(kind.rawValue)")
    }
}

struct AIDisclosureView: View {
    @State private var showsDetails = false

    var body: some View {
        Button { showsDetails = true } label: {
            Text(AppText.text("legal.ai.short"))
                .font(.caption2)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 6)
        }
        .buttonStyle(.plain)
        .accessibilityHint(AppText.text("legal.ai.hint"))
        .accessibilityIdentifier("chat.aiDisclosure")
        .sheet(isPresented: $showsDetails) {
            NavigationStack {
                LegalDocumentView(kind: .disclaimer)
                    .toolbar { ToolbarItem(placement: .topBarTrailing) {
                        Button(AppText.text("workspace.close")) { showsDetails = false }
                    } }
            }
        }
    }
}

/// Explains the mode before enabling it; it is not a per-request authorization.
struct AgentDisclosureView: View {
    let model: ModelInfo
    let onEnable: () -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                if model.isPocket03B {
                    Label(AppText.text("agent.pocketToolNotice"), systemImage: "info.circle")
                        .font(.callout)
                }
                Section {
                    Text(AppText.text("legal.agent.network"))
                    Text(AppText.text("legal.agent.memory"))
                    Text(AppText.text("legal.agent.choice"))
                }
                Section {
                    NavigationLink(LegalDocumentKind.privacy.title) { LegalDocumentView(kind: .privacy) }
                    NavigationLink(LegalDocumentKind.services.title) { LegalDocumentView(kind: .services) }
                }
                Section {
                    Button(AppText.text("legal.agent.enable")) {
                        onEnable()
                        dismiss()
                    }
                    .accessibilityIdentifier("agent.disclosure.enable")
                    Button(AppText.text("Cancel"), role: .cancel) { dismiss() }
                        .accessibilityIdentifier("agent.disclosure.cancel")
                }
            }
            .navigationTitle(AppText.text("legal.agent.title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .topBarTrailing) {
                Button(AppText.text("workspace.close")) { dismiss() }
            } }
        }
    }
}

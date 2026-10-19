import SwiftUI

enum AppSupportInfo {
    static var name: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleDisplayName") as? String ?? "MNN Chat"
    }

    static var version: String {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0"
        let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "1"
        return "\(version) (\(build))"
    }

    static func website(_ key: String) -> URL? {
        guard let value = Bundle.main.object(forInfoDictionaryKey: key) as? String,
              let url = URL(string: value), url.scheme == "https", url.host != nil else { return nil }
        return url
    }

    static var email: String? {
        guard let value = Bundle.main.object(forInfoDictionaryKey: "SupportEmail") as? String,
              !value.isEmpty, value.contains("@") else { return nil }
        return value
    }
}

struct PrivacyPolicyView: View {
    private let sections = ["local", "network", "permissions", "storage", "sharing"]

    var body: some View {
        List {
            Section {
                Text("privacy.summary")
                Text("privacy.date").font(.footnote).foregroundStyle(.secondary)
            }
            ForEach(sections, id: \.self) { section in
                Section(header: Text(AppText.text("privacy.\(section).title"))) {
                    Text(AppText.text("privacy.\(section).body"))
                }
            }
            if let url = AppSupportInfo.website("PrivacyPolicyURL") {
                Link("privacy.online", destination: url)
            }
            if let email = AppSupportInfo.email {
                Section(header: Text("privacy.contact")) {
                    Text(email).textSelection(.enabled)
                }
            }
        }
        .navigationTitle("settings.button.privacy")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct OpenSourceLicensesView: View {
    private var notices: String {
        guard let url = Bundle.module.url(forResource: "OpenSourceLicenses", withExtension: "txt"),
              let text = try? String(contentsOf: url, encoding: .utf8) else {
            return AppText.text("licenses.unavailable")
        }
        return text
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("licenses.models").font(.callout)
                Text(notices).font(.footnote).textSelection(.enabled)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
        }
        .navigationTitle("settings.button.licenses")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct AppSupportView: View {
    @State private var description = ""

    private var report: String {
        "\(AppSupportInfo.name) \(AppSupportInfo.version)\niOS \(UIDevice.current.systemVersion)\n\n\(description)"
    }

    private var mailURL: URL? {
        guard let email = AppSupportInfo.email else { return nil }
        var url = URLComponents()
        url.scheme = "mailto"
        url.path = email
        url.queryItems = [URLQueryItem(name: "subject", value: AppSupportInfo.name),
                          URLQueryItem(name: "body", value: report)]
        return url.url
    }

    var body: some View {
        List {
            Section(header: Text("support.faq")) {
                ForEach(["download", "offline", "slow", "delete"], id: \.self) { item in
                    DisclosureGroup(AppText.text("support.\(item).question")) {
                        Text(AppText.text("support.\(item).answer"))
                            .foregroundStyle(.secondary)
                    }
                }
            }
            Section(header: Text("support.describe"), footer: Text("support.reportPrivacy")) {
                TextEditor(text: $description)
                    .frame(minHeight: 120)
                    .accessibilityLabel(Text("support.describe"))
                ShareLink(item: report) {
                    Label("support.share", systemImage: "square.and.arrow.up")
                        .foregroundStyle(description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Color.secondary : Color.accentColor)
                }
                .disabled(description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                if let url = mailURL {
                    Link("support.send", destination: url)
                        .disabled(description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                if let url = AppSupportInfo.website("SupportURL") {
                    Link("support.website", destination: url)
                }
            }
        }
        .navigationTitle("settings.button.help")
        .navigationBarTitleDisplayMode(.inline)
    }
}

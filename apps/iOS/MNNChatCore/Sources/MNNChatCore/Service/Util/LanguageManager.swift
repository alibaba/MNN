
//
//  LanguageManager.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/1.
//

import Combine
import ExyteChat
import Foundation

public final class LanguageManager: ObservableObject {
    public static let shared = LanguageManager()

    var languageOptions: [String: String] {
        ["en": AppText.text("language.english"),
         "zh-Hans": AppText.text("language.simplifiedChinese")]
    }

    private let languageKey = "AppLanguage"

    @Published public private(set) var currentLanguage: String

    private init() {
        let saved = UserDefaults.standard.string(forKey: languageKey)
        let system = Locale.preferredLanguages.first ?? "en"
        currentLanguage = saved.flatMap { ["en", "zh-Hans"].contains($0) ? $0 : nil }
            ?? (system.hasPrefix("zh") ? "zh-Hans" : "en")
        ChatTextLocalization.useLanguage(currentLanguage)
    }

    func applyLanguage(_ code: String) {
        guard ["en", "zh-Hans"].contains(code), code != currentLanguage else { return }
        ChatTextLocalization.useLanguage(code)
        UserDefaults.standard.set(code, forKey: languageKey)
        UserDefaults.standard.set([code], forKey: "AppleLanguages")
        currentLanguage = code

        NotificationCenter.default.post(name: Notification.Name("LanguageChanged"), object: nil)
    }
}

enum AppText {
    static func text(_ key: String, defaultValue: String? = nil, comment: String = "") -> String {
        ChatTextLocalization.text(key, defaultValue: defaultValue, comment: comment)
    }
}


//
//  Color+Extension.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/1.
//

import Foundation

class LanguageManager {
    static let shared = LanguageManager()
    
    private let languageKey = "AppLanguage"
    
    var currentLanguage: String {
        get {
            return UserDefaults.standard.string(forKey: languageKey) ?? getSystemLanguage()
        }
        set {
            UserDefaults.standard.set(newValue, forKey: languageKey)
            UserDefaults.standard.synchronize()
        }
    }
    
    private func getSystemLanguage() -> String {
        let preferredLanguage = Locale.preferredLanguages.first ?? "en"
        if preferredLanguage.starts(with: "zh") {
            return "简体中文"
        } else {
            return "English"
        }
    }
    
    func applyLanguage(_ language: String) {
        currentLanguage = language
        
        let code = language == "简体中文" ? "zh-Hans" : "en"
        UserDefaults.standard.set([code], forKey: "AppleLanguages")
        UserDefaults.standard.synchronize()
        
        NotificationCenter.default.post(name: Notification.Name("LanguageChanged"), object: nil)
    }
}

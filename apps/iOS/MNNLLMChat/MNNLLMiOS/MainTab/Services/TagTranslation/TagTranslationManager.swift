//
//  TagTranslationManager.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/4.
//

import Foundation

// MARK: - 标签翻译管理器
class TagTranslationManager {
    static let shared = TagTranslationManager()
    private var tagTranslations: [String: String] = [:]
    
    private init() {}
    
    func loadTagTranslations(_ translations: [String: String]) {
        tagTranslations = translations
    }
    
    func getLocalizedTag(_ tag: String) -> String {
        let currentLanguage = LanguageManager.shared.currentLanguage
        let isChineseLanguage = currentLanguage == "简体中文"
        
        if isChineseLanguage, let translation = tagTranslations[tag] {
            return translation
        }
        return tag
    }
}

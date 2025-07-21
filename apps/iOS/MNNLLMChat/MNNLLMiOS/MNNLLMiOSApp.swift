//
//  MNNLLMiOSApp.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2024/12/26.
//

import SwiftUI

@main
struct MNNLLMiOSApp: App {
    
    init() {
        UIView.appearance().overrideUserInterfaceStyle = .light
        
        let savedLanguage = LanguageManager.shared.currentLanguage
        UserDefaults.standard.set([savedLanguage], forKey: "AppleLanguages")
        UserDefaults.standard.synchronize()
    }
    
    var body: some Scene {
        WindowGroup {
            MainTabView()
        }
    }
}

//
//  MNNLLMiOSApp.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2024/12/26.
//

import Darwin
import MNNChatCore
import SwiftUI

@main
struct MNNLLMiOSApp: App {
    /// DEBUG-only: apply Metal/LLM env overrides from
    /// Documents/mnn_env.json ({"MNN_METAL_FOO": "0", ...}), pushed via
    /// devicectl. Lets us A/B engine switches on device without rebuilding.
    /// Runs before any model can create a Metal runtime. (Validated: writes
    /// env_loader_ran.txt with the applied pairs.)
    private static func applyEnvOverridesFromDeviceFile() {
        #if DEBUG
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let url = documents.appendingPathComponent("mnn_env.json")
        guard let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            return
        }
        for (key, value) in dict {
            // The container path changes per install, so the pushed file may
            // use $DOCUMENTS as a placeholder (e.g. for the profile CSV path).
            let expanded = value.replacingOccurrences(of: "$DOCUMENTS", with: documents.path)
            setenv(key, expanded, 1)
        }
        #endif
    }

    init() {
        // DFlash currently requires command replay to be disabled. MetalEnv
        // snapshots environment variables on first use, so this must happen
        // before any model creates a Metal runtime.
        setenv("MNN_METAL_DISABLE_REPLAY", "1", 1)

        // Log every Metal pipeline compilation (kernel name + macros) so
        // device syslogs show whether a model load is progressing or hung.
        #if DEBUG
        setenv("MNN_METAL_PIPELINE_INFO", "1", 0)
        #endif

        // Last, so the pushed file can override even the hardcoded entries
        // (this is the point of the on-device A/B harness).
        Self.applyEnvOverridesFromDeviceFile()


        MNNChatConfiguration.install(.classic)

        UIView.appearance().overrideUserInterfaceStyle = .light

        let savedLanguage = LanguageManager.shared.currentLanguage
        UserDefaults.standard.set([savedLanguage], forKey: "AppleLanguages")
        UserDefaults.standard.synchronize()
    }

    var body: some Scene {
        WindowGroup {
            #if DEBUG
            if CommandLine.arguments.contains("--preset-asr-silent-probe") {
                PresetASRSilentProbeView()
            } else {
                MainTabView()
            }
            #else
            MainTabView()
            #endif
        }
    }
}

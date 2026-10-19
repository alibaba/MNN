//
//  MNNPocketApp.swift
//  MNNPocket
//
//  Created by 游薪渝(揽清) on 2024/12/26.
//

import Darwin
import MNNChatCore
import SwiftUI

@main
struct MNNPocketApp: App {
    init() {
        // DFlash currently requires command replay to be disabled. MetalEnv
        // snapshots environment variables on first use, so this must happen
        // before any model creates a Metal runtime. Keep a launch-time override
        // available for one-switch-at-a-time device diagnostics.
        setenv("MNN_METAL_DISABLE_REPLAY", "1", 0)

        // Diagnostic: log every Metal pipeline compilation (kernel name +
        // macros) so device console captures show exactly which kernels the
        // runtime built in good vs all-Inf sessions.
        setenv("MNN_METAL_PIPELINE_INFO", "1", 0)

        // Bisection: the first corrupted tensor is the Q4 GEMV output
        // (wrong dequant scale). Ruled out across the 0.3B/0.8B models:
        // W16 specialization, dual-stream SPLIT_K, W16_DS, GATE_UP_FUSION.
        // Next: force prefill off the in-shader dequant body onto the
        // outer-dequant + fp route (completely bypasses in-kernel W4
        // unpacking for area > 1).
        setenv("MNN_METAL_PREFILL_INSHADER_DEQUANT_SGMATRIX", "0", 0)

        MNNChatConfiguration.install(.pocket)

        // The Pocket app uses a light theme (white background, dark text).
        UIView.appearance().overrideUserInterfaceStyle = .light

        let savedLanguage = LanguageManager.shared.currentLanguage
        UserDefaults.standard.set([savedLanguage], forKey: "AppleLanguages")
        UserDefaults.standard.synchronize()

#if DEBUG
        if CommandLine.arguments.contains("--native-probe") {
            NativeProbe.run()
        }
#endif
    }

    var body: some Scene {
        WindowGroup {
            if CommandLine.arguments.contains("--native-probe") {
                Text("Native Metal probe running")
            } else if CommandLine.arguments.contains("--preset-asr-silent-probe") {
                PresetASRSilentProbeView()
            } else {
                PocketAppView()
            }
        }
    }
}

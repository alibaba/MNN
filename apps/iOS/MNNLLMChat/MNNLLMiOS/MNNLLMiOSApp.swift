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
            if CommandLine.arguments.contains("--preset-asr-silent-probe") {
                PresetASRSilentProbeView()
            } else {
                MainTabView()
            }
        }
    }
}

private struct PresetASRSilentProbeView: View {
    @State private var status = "X-ASR silent probe"

    var body: some View {
        Text(status).task {
            status = await Task.detached(priority: .userInitiated) {
                await Self.runProbe()
            }.value
        }
    }

    private static func runProbe() async -> String {
        let reportURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("qwen-xasr-device-probe.json")
        let audioURL = Bundle.main.resourceURL?
            .appendingPathComponent("LocalModel/preset_audio/audio (1).wav")
        var report: [String: Any] = [
            "runtime": "sherpa-mnn",
            "model": "x-asr-zipformer2",
            "precision": "mnn-weight-int8-block64",
            "backend": "cpu",
            "threads": 2,
        ]
        let service = PresetZipformerASRService()
        do {
            guard let audioURL else {
                throw PresetASRError.missingAsset("LocalModel/preset_audio/audio (1).wav")
            }
            let recognition = try await service.recognize(
                audioURL: audioURL,
                expectedAudioSHA256: "2c9c3ab83563e058ed46e067e716beeeca8d3b50a2cd0e622dc6437714647b58"
            )
            report["transcript"] = recognition.transcript
            report["audio_seconds"] = recognition.audioDurationSeconds
            report["initialization_seconds"] = recognition.modelInitializationSeconds
            report["inference_seconds"] = recognition.inferenceSeconds
            report["rtf"] = recognition.realTimeFactor
            report["realtime_multiple"] = recognition.realTimeMultiple
        } catch {
            report["error"] = error.localizedDescription
        }
        service.shutdown()
        if let data = try? JSONSerialization.data(
            withJSONObject: report, options: [.prettyPrinted, .sortedKeys]
        ) {
            try? data.write(to: reportURL, options: .atomic)
        }
        return report["error"] == nil ? "X-ASR silent probe complete" : "X-ASR silent probe failed"
    }
}

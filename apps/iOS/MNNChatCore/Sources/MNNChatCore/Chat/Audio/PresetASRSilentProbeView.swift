//
//  PresetASRSilentProbeView.swift
//  MNNChatCore
//
//  Shared `--preset-asr-silent-probe` harness view, previously duplicated in
//  both app entry files. Runs the bundled X-ASR model over a preset audio
//  file and writes a JSON report to the documents directory.
//

import Foundation
import SwiftUI

public struct PresetASRSilentProbeView: View {
    @State private var status = "X-ASR silent probe"

    public init() {}

    public var body: some View {
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
            "backend": "metal",
            "threads": 2,
        ]
        let service = PresetZipformerASRService()
        do {
            guard let audioURL else {
                throw PresetASRError.missingAsset("LocalModel/preset_audio/audio (1).wav")
            }
            let recognition = try await service.recognize(
                audioURL: audioURL,
                expectedAudioSHA256: "b112c5b9b4503e59b14102be18bba6cc0cb9cd44a86af87c2b75a31e1f06f066"
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

//
//  PresetPrompts.swift
//  MNNLLMiOS
//

import Foundation

struct PresetPrompt {
    let title: String
    let icon: String
    let text: String
    let imageBundlePath: String?
    let audioBundlePath: String?
    let audioSHA256: String?

    init(
        title: String,
        icon: String,
        text: String,
        imageBundlePath: String?,
        audioBundlePath: String? = nil,
        audioSHA256: String? = nil
    ) {
        self.title = title
        self.icon = icon
        self.text = text
        self.imageBundlePath = imageBundlePath
        self.audioBundlePath = audioBundlePath
        self.audioSHA256 = audioSHA256
    }

    var isMultimodal: Bool { imageBundlePath != nil }
    var isASRAudio: Bool { audioBundlePath != nil }
}

enum PresetPrompts {
    static func all() -> [PresetPrompt] {
        var presets: [PresetPrompt] = []

        if let text512 = loadText("512.txt") {
            presets.append(PresetPrompt(title: "512 tokens", icon: "doc.text", text: text512, imageBundlePath: nil))
        }
        if let text1024 = loadText("1024.txt") {
            presets.append(PresetPrompt(title: "1024 tokens", icon: "doc.text.fill", text: text1024, imageBundlePath: nil))
        }

        presets.append(imagePreset(
            file: "cat.jpg",
            title: NSLocalizedString("preset.cat", value: "图片·小猫", comment: ""),
            text: NSLocalizedString(
                "preset.catPrompt",
                value: "请仔细观察这张图片，然后用完整的几句话描述：画面的主体是什么、它的颜色和姿态、所处的背景环境，以及这张照片整体的氛围。",
                comment: ""
            )
        ))
        presets.append(imagePreset(
            file: "shapes.jpg",
            title: NSLocalizedString("preset.shapes", value: "图片·形状", comment: ""),
            text: NSLocalizedString(
                "preset.shapesPrompt",
                value: "请仔细观察这张图片，逐一说明图中每个几何图形的形状、颜色和大致位置，最后总结一共有几个图形。",
                comment: ""
            )
        ))

        if let audio = audioPreset(
            file: "audio (1).wav",
            title: NSLocalizedString("preset.audio", value: "音频·杭州美食", comment: "")
        ) {
            presets.append(audio)
        }

        presets.append(PresetPrompt(
            title: NSLocalizedString("preset.code", value: "代码题", comment: ""),
            icon: "chevron.left.forwardslash.chevron.right",
            text: NSLocalizedString(
                "preset.codePrompt",
                value: "请用 C++ 实现一个 O(1) 的 LRU 缓存，包含 get 和 put 接口，给出完整代码并解释设计思路。",
                comment: ""
            ),
            imageBundlePath: nil
        ))
        return presets
    }

    private static func imagePreset(file: String, title: String, text: String) -> PresetPrompt {
        PresetPrompt(title: title, icon: "photo", text: text, imageBundlePath: bundlePath("LocalModel/preset_images/\(file)"))
    }

    private static func audioPreset(file: String, title: String) -> PresetPrompt? {
        guard let path = bundlePath("LocalModel/preset_audio/\(file)") else { return nil }
        return PresetPrompt(
            title: title,
            icon: "waveform",
            text: "",
            imageBundlePath: nil,
            audioBundlePath: path,
            audioSHA256: "2c9c3ab83563e058ed46e067e716beeeca8d3b50a2cd0e622dc6437714647b58"
        )
    }

    private static func bundlePath(_ relative: String) -> String? {
        guard let root = Bundle.main.resourcePath else { return nil }
        let path = (root as NSString).appendingPathComponent(relative)
        return FileManager.default.fileExists(atPath: path) ? path : nil
    }

    private static func loadText(_ file: String) -> String? {
        guard let path = bundlePath("LocalModel/\(file)") else { return nil }
        let text = try? String(contentsOfFile: path, encoding: .utf8)
        guard let text, !text.isEmpty else { return nil }
        return text
    }
}

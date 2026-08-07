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

    var isMultimodal: Bool { imageBundlePath != nil }
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

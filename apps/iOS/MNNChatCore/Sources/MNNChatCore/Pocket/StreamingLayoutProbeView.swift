#if DEBUG
import ExyteChat
import SwiftUI
import UIKit

/// Opt-in rendering replay; no model or saved conversation is involved.
struct StreamingLayoutProbeView: View {
    private let legacy = CommandLine.arguments.contains("--legacy-renderer")
    @State private var messageId = UUID().uuidString
    @State private var text = ""
    @State private var complete = false
    @State private var finished = false
    @State private var frames = StreamingFrameProbe()
    @State private var outputEnded: TimeInterval = 0

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Text(legacy ? "Streaming replay: previous renderer" : "Streaming replay: updated renderer")
                        .font(.headline)
                    if legacy {
                        MarkdownTextViewWrapper(text: text)
                    } else {
                        StreamingMarkdownTextView(text: text, messageId: messageId,
                            isStreaming: !finished, isOutputComplete: complete,
                            onReveal: frames.recordReveal)
                    }
                    Color.clear.frame(height: 1).id("probe-bottom")
                }
                .padding(16)
                .background {
                    GeometryReader { geometry in
                        Color.clear.preference(key: StreamingProbeHeight.self, value: geometry.size.height)
                    }
                }
            }
            .onPreferenceChange(StreamingProbeHeight.self) { _ in
                proxy.scrollTo("probe-bottom", anchor: .bottom)
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSNotification.Name("StreamingAnimationCompleted"))) { note in
            guard note.userInfo?["messageId"] as? String == messageId else { return }
            finish()
        }
        .onChange(of: text) { _, value in
            if legacy { frames.recordReveal(value.utf8.count) }
        }
        .task {
            frames.start()
            let characters = Array(Self.sample)
            for start in stride(from: 0, to: characters.count, by: 32) {
                guard !Task.isCancelled else { return }
                text.append(contentsOf: characters[start..<min(start + 32, characters.count)])
                try? await Task.sleep(for: .milliseconds(80))
            }
            outputEnded = ProcessInfo.processInfo.systemUptime
            complete = true
            if legacy { finish() }
            else {
                try? await Task.sleep(for: .seconds(4))
                if !finished { finish(timedOut: true) }
            }
        }
        .onDisappear { frames.stop() }
    }

    private func finish(timedOut: Bool = false) {
        guard !finished else { return }
        finished = true
        frames.stop()
        let values = frames.gaps.sorted()
        let p95 = values.isEmpty ? 0 : values[min(values.count - 1, Int(Double(values.count) * 0.95))]
        let revealGaps = zip(frames.reveals.dropFirst(), frames.reveals).map { $0 - $1 }.sorted()
        let result: [String: Any] = [
            "renderer": legacy ? "previous" : "updated", "frames": values.count,
            "frameGapP95Ms": p95 * 1000, "frameGapMaxMs": (values.last ?? 0) * 1000,
            "framesOver50Ms": values.filter { $0 > 0.05 }.count,
            "revealTailMs": max(0, ProcessInfo.processInfo.systemUptime - outputEnded) * 1000,
            "inputUTF8Bytes": text.utf8.count, "expectedUTF8Bytes": Self.sample.utf8.count,
            "revealedUTF8Bytes": frames.revealedCount, "revealUpdates": frames.reveals.count,
            "revealIntervalMedianMs": revealGaps.isEmpty ? 0 : revealGaps[revealGaps.count / 2] * 1000,
            "revealIntervalP95Ms": revealGaps.isEmpty ? 0 : revealGaps[min(revealGaps.count - 1, Int(Double(revealGaps.count) * 0.95))] * 1000,
            "completed": complete && !timedOut && frames.revealedCount == Self.sample.utf8.count
        ]
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let name = legacy ? "streaming-probe-previous.json" : "streaming-probe-updated.json"
        if let data = try? JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: documents.appendingPathComponent(name), options: .atomic)
        }
        print("[StreamingProbe] \(result)")
    }

    private static var sample: String {
        (1...10).map { index in
            """
            ## 第 \(index) 段：杭州与端侧 AI

            杭州以西湖、龙井茶和历史街区闻名。端侧模型在手机本地生成回答，文字应连续出现，长段落也应保持稳定。这里检查中文、英文、数字、**重点文字**和 emoji 👩🏽‍💻，以及分批收到输出时的排版。

            - 湖边散步，欣赏湖光山色。
            - 阅读文档，提取有用的信息。

            ```python
            def greet(name):
                return f"你好，{name}!"
            print(greet("杭州"))
            ```


            """
        }.joined()
    }
}

private struct StreamingProbeHeight: PreferenceKey {
    static let defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}

private final class StreamingFrameProbe: NSObject {
    private var link: CADisplayLink?
    private var last: CFTimeInterval?
    private(set) var gaps: [TimeInterval] = []
    private(set) var reveals: [TimeInterval] = []
    private(set) var revealedCount = 0

    func recordReveal(_ count: Int) {
        guard count > revealedCount else { return }
        revealedCount = count
        reveals.append(ProcessInfo.processInfo.systemUptime)
    }

    func start() {
        stop()
        gaps = []
        last = nil
        let link = CADisplayLink(target: self, selector: #selector(tick(_:)))
        link.add(to: .main, forMode: .common)
        self.link = link
    }

    func stop() { link?.invalidate(); link = nil }

    @objc private func tick(_ link: CADisplayLink) {
        if let last { gaps.append(link.timestamp - last) }
        last = link.timestamp
    }
}

#endif

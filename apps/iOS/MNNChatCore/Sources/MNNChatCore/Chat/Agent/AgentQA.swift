#if DEBUG
import CMNNInference
import Foundation
import UIKit

// Launch-argument-only device test driver; normal app launches never enter it.
enum AgentQA {
    static let modelName: String = {
        let args = CommandLine.arguments
        guard let index = args.firstIndex(of: "--agent-qa-model"), index + 1 < args.count else { return "" }
        return args[index + 1]
    }()
    static var enabled: Bool { !modelName.isEmpty }
    static var started = false
    static var currentCase = "setup"
    static var results: [[String: Any]] = []
    static var settings: [String: Any] = [:]
    static var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }
    static var traceURL: URL { documents.appendingPathComponent("agent-qa-\(modelName).jsonl") }
    static var resultURL: URL { documents.appendingPathComponent("agent-qa-\(modelName).json") }

    static func record(_ event: String, _ fields: [String: Any] = [:]) {
        guard enabled else { return }
        var entry = fields
        entry["event"] = event
        entry["case"] = currentCase
        entry["timestamp"] = ISO8601DateFormatter().string(from: Date())
        guard var data = try? JSONSerialization.data(withJSONObject: entry, options: [.sortedKeys]) else { return }
        data.append(0x0a)
        if !FileManager.default.fileExists(atPath: traceURL.path) {
            try? data.write(to: traceURL)
        } else if let file = try? FileHandle(forWritingTo: traceURL) {
            defer { try? file.close() }
            _ = try? file.seekToEnd()
            try? file.write(contentsOf: data)
        }
        print("[AgentQA] \(currentCase) \(event)")
    }

    static func saveResults(completed: Bool) {
        let value: [String: Any] = ["model": modelName, "completed": completed, "cases": results,
            "settings": settings,
            "device": UIDevice.current.model, "ios": UIDevice.current.systemVersion]
        if let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: resultURL, options: .atomic)
        }
    }

    @MainActor
    static func run(workspace: ChatWorkspaceModel) async {
        guard enabled, !started else { return }
        started = true
        try? Data().write(to: traceURL, options: .atomic)
        UIApplication.shared.isIdleTimerDisabled = true
        defer { UIApplication.shared.isIdleTimerDisabled = false }
        record("load_start")
        for _ in 0..<400 {
            if let chat = workspace.currentChat, chat.isModelLoaded, !chat.isProcessing { break }
            if workspace.currentChat?.loadFailed == true { break }
            try? await Task.sleep(for: .milliseconds(500))
        }
        guard let chat = workspace.currentChat, chat.modelInfo.modelName == modelName,
              chat.isModelLoaded, !chat.isProcessing else {
            record("load_failed"); saveResults(completed: false); return
        }
        let configFile = documents.appendingPathComponent("MNNConfigs/\(modelName)/custom_config.json")
        let originalConfig = try? Data(contentsOf: configFile)
        let originalThinking = chat.isThinkingModeEnabled
        let originalContext = chat.isConversationContextEnabled
        let originalLimit = chat.contextLimitTokens
        let config = ModelConfigManager(modelPath: chat.modelInfo.localPath)
        settings = ["backend": config.readBackendType(), "precision": config.readPrecision(),
            "use_mmap": config.readUseMmap(), "context_limit": 4096, "max_new_tokens": 512,
            "thinking": false, "sampler": "greedy"]
        if let disk = try? FileManager.default.attributesOfFileSystem(forPath: documents.path),
           let free = disk[.systemFreeSize] as? NSNumber { settings["free_disk_bytes_at_start"] = free }
        let originalMax = config.readMaxNewTokens()
        defer {
            chat.isThinkingModeEnabled = originalThinking
            chat.configureThinkingMode()
            chat.setConversationContextEnabled(originalContext)
            chat.setContextTokenLimit(originalLimit)
            chat.llm?.setConfigWithJSONString("{\"max_new_tokens\":\(originalMax)}")
            if let originalConfig { try? originalConfig.write(to: configFile, options: .atomic) }
            else { try? FileManager.default.removeItem(at: configFile) }
        }
        AgentStore.shared?.clearMemories()
        AgentStore.shared?.clearSkills()
        chat.isThinkingModeEnabled = false
        chat.configureThinkingMode()
        chat.setConversationContextEnabled(true)
        chat.setContextTokenLimit(4096)
        config.updateMaxNewTokens(512)
        chat.llm?.setConfigWithJSONString("{\"max_new_tokens\":512,\"sampler_type\":\"greedy\"}")
        record("loaded", ["path": chat.modelInfo.localPath, "supportsThinking": chat.supportsThinkingMode])

        let cases: [(String, String, Bool)] = [
            ("time", "现在准确的本地时间是几点几分？请查询当前时间，并写出时区。", true),
            ("calculator", "请计算 (137×29+863)÷8，给出准确结果。", true),
            ("shopping", "3件商品每件399元，另有2件配件每件89元，商品和配件合计打85折，最后加12元运费。一共应付多少元？", true),
            ("browse", "请打开 https://example.com ，告诉我网页标题，并用一句中文概括它的用途。", true),
            ("search", "请联网搜索阿里巴巴 MNN 的官方 GitHub 仓库，给出仓库链接，并用一句话说明它是什么。", true),
            ("search_read", "请先搜索阿里巴巴 MNN 的官方 GitHub 仓库，再打开搜索到的仓库页面。根据读到的内容列出3个特点，并附链接；读不到时请明确说明。", true),
            ("context_seed", "这次计算有两笔费用：A是37元，B是58元。先只回复已记录。", true),
            ("context_followup", "把上面两笔费用相加后打九折，应付多少元？", false),
            ("unavailable_page", "请打开 https://agent-probe.invalid/report 告诉我报告标题；如果网页无法访问，请明确说明，别猜内容。", true),
            ("memory_write", "请记住：本次测试的项目代号是松果73。", true),
            ("memory_recall", "我之前让你记住的项目代号是什么？只回答代号。", true)
        ]
        for (id, prompt, fresh) in cases {
            currentCase = id
            if fresh {
                guard workspace.newConversation() else { record("reset_failed"); return }
                chat.setAgentMode(true)
            }
            guard chat.isAgentMode else { record("agent_mode_failed"); return }
            let previousIDs = Set(chat.messages.map(\.id))
            let start = ProcessInfo.processInfo.systemUptime
            var firstAnswerSeconds: Double?
            record("case_start", ["prompt": prompt, "newConversation": fresh])
            chat.sendUserPrompt(prompt)
            for _ in 0..<480 {
                try? await Task.sleep(for: .milliseconds(500))
                if firstAnswerSeconds == nil, chat.messages.contains(where: {
                    !previousIDs.contains($0.id) && $0.user.id == "2" && !$0.text.isEmpty
                }) { firstAnswerSeconds = ProcessInfo.processInfo.systemUptime - start }
                if !chat.isProcessing { break }
            }
            guard !chat.isProcessing else {
                record("case_timeout"); saveResults(completed: false); return
            }
            try? await Task.sleep(for: .milliseconds(300))
            let answers = chat.messages.filter { !previousIDs.contains($0.id) && $0.user.id == "2" }
            let steps = answers.flatMap(\.toolSteps)
            let stepJSON = (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(steps))) ?? []
            var result: [String: Any] = ["id": id, "prompt": prompt,
                "response": answers.map(\.text).joined(separator: "\n"),
                "thinking": answers.compactMap(\.thinkText).joined(separator: "\n"),
                "steps": stepJSON, "elapsed_seconds": ProcessInfo.processInfo.systemUptime - start,
                "input_tokens_last_pass": chat.contextUsage.inputTokens,
                "generated_tokens_last_pass": chat.contextUsage.generatedTokens,
                "available_memory_bytes": chat.contextUsage.availableMemoryBytes ?? 0,
                "footprint_bytes": chat.contextUsage.footprintBytes ?? 0,
                "memories": AgentStore.shared?.allMemories().map(\.content) ?? []]
            if let firstAnswerSeconds { result["first_visible_answer_seconds"] = firstAnswerSeconds }
            results.append(result)
            record("case_complete", result)
            chat.saveHistoryImmediately()
            saveResults(completed: false)
        }
        currentCase = "finished"
        saveResults(completed: true)
        record("completed", ["cases": results.count])
    }
}

#endif

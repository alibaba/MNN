import AppKit
import Foundation
import WebKit
import UniformTypeIdentifiers
import CryptoKit

final class AppController: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate, WKScriptMessageHandler {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var rootView: NSView!
    private let titleLabel = NSTextField(labelWithString: "MNN Studio")
    private let statusLabel = NSTextField(labelWithString: "请选择模型")
    private let modelButton = NSButton(title: "选择模型…", target: nil, action: nil)
    private let embeddingButton = NSButton(title: "向量模型…", target: nil, action: nil)
    private var engine: Process?
    private var logHandle: FileHandle?
    private let port = 18957
    private var apiKey = ""
    private var activeModel = ""
    private var modelConfig: String?
    private var embeddingConfig: String?
    private var generation = 0

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        makeMenu()
        makeWindow()
        let arguments = CommandLine.arguments
        if let index = arguments.firstIndex(of: "--config"), index + 1 < arguments.count {
            modelConfig = arguments[index + 1]
        } else {
            modelConfig = UserDefaults.standard.string(forKey: "modelConfig")
        }
        embeddingConfig = UserDefaults.standard.string(forKey: "embeddingConfig")
        if let path = modelConfig, FileManager.default.fileExists(atPath: path) {
            startEngine(config: path)
        } else {
            showWelcome("选择已导出的 MNN 模型配置文件，开始本地对话。")
        }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func makeMenu() {
        let main = NSMenu()
        let application = NSMenuItem()
        main.addItem(application)
        let menu = NSMenu()
        menu.addItem(withTitle: "关于 MNN Studio", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        menu.addItem(.separator())
        menu.addItem(withTitle: "退出 MNN Studio", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        application.submenu = menu
        let serviceItem = NSMenuItem()
        serviceItem.title = "服务"
        main.addItem(serviceItem)
        let serviceMenu = NSMenu(title: "服务")
        let copyKey = serviceMenu.addItem(withTitle: "复制 API Key", action: #selector(copyAPIKey), keyEquivalent: "")
        copyKey.target = self
        let removeEmbedding = serviceMenu.addItem(withTitle: "移除向量模型", action: #selector(clearEmbedding), keyEquivalent: "")
        removeEmbedding.target = self
        serviceItem.submenu = serviceMenu
        NSApp.mainMenu = main
    }

    private func makeWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 820),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "MNN Studio"
        window.minSize = NSSize(width: 760, height: 520)
        window.center()
        window.delegate = self
        let root = NSView()
        rootView = root
        root.wantsLayer = true
        root.layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
        window.contentView = root

        let bar = NSView()
        bar.translatesAutoresizingMaskIntoConstraints = false
        root.addSubview(bar)
        let title = titleLabel
        title.font = .systemFont(ofSize: 13, weight: .semibold)
        title.textColor = .labelColor
        statusLabel.font = .systemFont(ofSize: 11)
        statusLabel.textColor = .secondaryLabelColor
        modelButton.target = self
        modelButton.action = #selector(selectModel)
        embeddingButton.target = self
        embeddingButton.action = #selector(selectEmbedding)
        for view in [title, statusLabel, modelButton, embeddingButton] {
            view.translatesAutoresizingMaskIntoConstraints = false
            bar.addSubview(view)
        }
        let configuration = WKWebViewConfiguration()
        configuration.userContentController.add(self, name: "appearance")
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        webView.translatesAutoresizingMaskIntoConstraints = false
        root.addSubview(webView)
        NSLayoutConstraint.activate([
            bar.topAnchor.constraint(equalTo: root.topAnchor), bar.leadingAnchor.constraint(equalTo: root.leadingAnchor),
            bar.trailingAnchor.constraint(equalTo: root.trailingAnchor), bar.heightAnchor.constraint(equalToConstant: 52),
            title.leadingAnchor.constraint(equalTo: bar.leadingAnchor, constant: 20), title.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
            statusLabel.leadingAnchor.constraint(equalTo: title.trailingAnchor, constant: 18), statusLabel.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
            embeddingButton.trailingAnchor.constraint(equalTo: bar.trailingAnchor, constant: -16), embeddingButton.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
            modelButton.trailingAnchor.constraint(equalTo: embeddingButton.leadingAnchor, constant: -8), modelButton.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
            statusLabel.trailingAnchor.constraint(lessThanOrEqualTo: modelButton.leadingAnchor, constant: -12),
            webView.topAnchor.constraint(equalTo: bar.bottomAnchor), webView.leadingAnchor.constraint(equalTo: root.leadingAnchor),
            webView.trailingAnchor.constraint(equalTo: root.trailingAnchor), webView.bottomAnchor.constraint(equalTo: root.bottomAnchor)
        ])
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.name == "appearance", let values = message.body as? [String: String],
              let mode = values["mode"], let theme = values["resolved"],
              ["system", "light", "dark"].contains(mode), ["light", "dark"].contains(theme) else { return }
        let isDark = mode == "system" ? NSApp.effectiveAppearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua : theme == "dark"
        window.appearance = mode == "system" ? nil : NSAppearance(named: isDark ? .darkAqua : .aqua)
        rootView.layer?.backgroundColor = (isDark ? NSColor(calibratedWhite: 0.09, alpha: 1) :
            NSColor(calibratedRed: 0.94, green: 0.94, blue: 0.96, alpha: 1)).cgColor
        titleLabel.textColor = isDark ? .white : .labelColor
    }

    private func showWelcome(_ message: String) {
        let escaped = message.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;")
        let html = """
        <html><head><meta charset="utf-8"><style>
        body{background:#111216;color:#eeedf3;font:14px -apple-system,sans-serif;display:grid;place-items:center;height:100vh;margin:0}
        main{text-align:center;max-width:460px} .mark{display:grid;place-items:center;margin:auto auto 24px;width:62px;height:62px;
        border-radius:18px;background:#39334b;color:#c5bdff;font-size:28px}h1{font-size:27px;letter-spacing:-.04em}
        p{color:#a3a3b3;line-height:1.8} small{color:#747584}
        @media (prefers-color-scheme:light){body{background:#f7f7fa;color:#242431} .mark{background:#eae7f8;color:#6557bd}p{color:#666779}}
        </style></head><body><main><div class="mark">M</div><h1>欢迎使用 MNN Studio</h1>
        <p>\(escaped)</p><small>使用窗口右上角的“选择模型”打开 config.json</small></main></body></html>
        """
        webView.loadHTMLString(html, baseURL: nil)
    }

    @objc private func selectModel() {
        let panel = NSOpenPanel()
        panel.title = "选择 MNN 模型配置"
        panel.message = "选择模型目录中的 config.json。"
        panel.allowedContentTypes = [.json]
        panel.canChooseDirectories = false
        panel.beginSheetModal(for: window) { [weak self] response in
            guard response == .OK, let path = panel.url?.path, let self = self else { return }
            self.modelConfig = path
            UserDefaults.standard.set(path, forKey: "modelConfig")
            self.startEngine(config: path)
        }
    }

    @objc private func selectEmbedding() {
        let panel = NSOpenPanel()
        panel.title = "选择 embedding 模型配置"
        panel.message = "选择 embedding 模型目录中的 config.json；取消不会更改现有配置。"
        panel.allowedContentTypes = [.json]
        panel.canChooseDirectories = false
        panel.beginSheetModal(for: window) { [weak self] response in
            guard response == .OK, let path = panel.url?.path, let self = self else { return }
            self.embeddingConfig = path
            UserDefaults.standard.set(path, forKey: "embeddingConfig")
            if let config = self.modelConfig { self.startEngine(config: config) }
        }
    }

    @objc private func clearEmbedding() {
        embeddingConfig = nil
        UserDefaults.standard.removeObject(forKey: "embeddingConfig")
        if let config = modelConfig { startEngine(config: config) }
    }

    @objc private func copyAPIKey() {
        guard engine?.isRunning == true else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(apiKey, forType: .string)
    }

    private func dataDirectory() throws -> URL {
        let override = ProcessInfo.processInfo.environment["MNN_STUDIO_DATA_DIR"]
        let base = override.map { URL(fileURLWithPath: $0, isDirectory: true) } ??
            FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("MNN Studio", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    private func startEngine(config: String) {
        generation += 1
        let attempt = generation
        stopEngine()
        statusLabel.stringValue = "正在加载模型…"
        showWelcome("正在加载模型，请稍候。")
        guard let binary = Bundle.main.url(forResource: "mnncli", withExtension: nil) else {
            showError("应用内缺少 mnncli 程序。")
            return
        }
        do {
            let directory = try dataDirectory()
            let logs = directory.appendingPathComponent("logs", isDirectory: true)
            try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
            let logURL = logs.appendingPathComponent("server-\(UUID().uuidString).log")
            guard FileManager.default.createFile(atPath: logURL.path, contents: nil) else {
                showError("无法创建服务日志。")
                return
            }
            let handle = try FileHandle(forWritingTo: logURL)
            logHandle = handle
            apiKey = UUID().uuidString
            let directoryName = URL(fileURLWithPath: config).deletingLastPathComponent().lastPathComponent
            let digest = SHA256.hash(data: Data(config.utf8)).prefix(3).map { String(format: "%02x", $0) }.joined()
            activeModel = "\(directoryName)-\(digest)"
            let process = Process()
            process.executableURL = binary
            process.currentDirectoryURL = directory
            process.arguments = ["serve", activeModel, "--config", config, "--host", "127.0.0.1",
                                 "--port", String(port), "--api-key", apiKey]
            if let embedding = embeddingConfig, FileManager.default.fileExists(atPath: embedding) {
                process.arguments?.append(contentsOf: ["--embedding-config", embedding, "--embedding-model", "mnn-embedding"])
            }
            process.standardOutput = handle
            process.standardError = handle
            process.terminationHandler = { [weak self, weak process] _ in
                DispatchQueue.main.async {
                    guard let self = self, let process = process, self.engine === process else { return }
                    self.engine = nil
                    self.statusLabel.stringValue = "服务已停止"
                    self.showWelcome("服务未能继续运行。请检查模型文件和日志：\(logURL.path)")
                }
            }
            try process.run()
            engine = process
            pollReady(attempt: attempt, remaining: 180)
        } catch {
            showError("无法启动服务：\(error.localizedDescription)")
        }
    }

    private func pollReady(attempt: Int, remaining: Int) {
        guard attempt == generation, let process = engine, process.isRunning else { return }
        guard remaining > 0 else { stopEngine(); showError("模型加载超时，请检查模型文件和服务日志。"); return }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/status")!)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 1
        URLSession.shared.dataTask(with: request) { [weak self] data, response, _ in
            DispatchQueue.main.async {
                guard let self = self, attempt == self.generation, self.engine === process else { return }
                if let response = response as? HTTPURLResponse, response.statusCode == 200,
                   let data = data, let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   value["model"] as? String == self.activeModel {
                    self.statusLabel.stringValue = "本地服务已连接 · 127.0.0.1:\(self.port)"
                    var components = URLComponents(string: "http://127.0.0.1:\(self.port)/")!
                    components.queryItems = [URLQueryItem(name: "token", value: self.apiKey)]
                    self.webView.load(URLRequest(url: components.url!))
                } else {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { self.pollReady(attempt: attempt, remaining: remaining - 1) }
                }
            }
        }.resume()
    }

    private func stopEngine() {
        let old = engine
        engine = nil
        if old?.isRunning == true { old?.terminate(); old?.waitUntilExit() }
        logHandle = nil
    }

    private func showError(_ message: String) {
        statusLabel.stringValue = "启动失败"
        showWelcome(message)
    }

    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = navigationAction.request.url else { decisionHandler(.cancel); return }
        if url.scheme == "about" || (url.scheme == "http" && url.host == "127.0.0.1" && url.port == port) {
            decisionHandler(.allow)
        } else {
            decisionHandler(.cancel)
        }
    }

    func windowWillClose(_ notification: Notification) { NSApp.terminate(nil) }
    func applicationWillTerminate(_ notification: Notification) { generation += 1; stopEngine() }
}

let app = NSApplication.shared
let controller = AppController()
app.delegate = controller
app.run()

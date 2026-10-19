import ArgumentParser
import CoreML
import Foundation

import Generation
import Models

@available(iOS 16.2, macOS 13.1, *)
struct TransformersCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Run text generation on a Core ML language model",
        version: "0.0.1"
    )

    @Argument(help: "Input text")
    var prompt: String

    @Argument(help: "Path to Core ML mlpackage model")
    var modelPath: String = "./model.mlpackage"

    @Option(help: "Maximum amount of tokens the model should generate")
    var maxLength: Int = 50

    @Option(help: "Compute units to load model with {all,cpuOnly,cpuAndGPU,cpuAndNeuralEngine}")
    var computeUnits: ComputeUnits = .cpuAndGPU

    func generate(model: LanguageModel, config: GenerationConfig, prompt: String, printOutput: Bool = true) {
        let semaphore = DispatchSemaphore(value: 0)
        Task.init { [config] in
            defer { semaphore.signal() }
            var tokensReceived = 0
            var previousIndex: String.Index? = nil
            let begin = Date()
            do {
                try await model.generate(config: config, prompt: prompt) { inProgressGeneration in
                    tokensReceived += 1
                    let response = inProgressGeneration.replacingOccurrences(of: "\\n", with: "\n")
                    if printOutput {
                        print(response[(previousIndex ?? response.startIndex)...], terminator: "")
                        fflush(stdout)
                    }
                    previousIndex = response.endIndex
                }
                let completionTime = Date().timeIntervalSince(begin)
                let tps = Double(tokensReceived) / completionTime
                if printOutput {
                    print("")
                    print("\(tps.formatted("%.2f")) tokens/s, total time: \(completionTime.formatted("%.2f"))s")
                }
            } catch {
                print("Error \(error)")
            }
        }
        semaphore.wait()
    }

    func compile(at url: URL) throws -> URL {
        #if os(watchOS)
        fatalError("Model compilation is not supported on watchOS")
        #else
        if url.pathExtension == "mlmodelc" { return url }
        print("Compiling model \(url)")
        return try MLModel.compileModel(at: url)
        #endif
    }

    func run() throws {
        let url = URL(filePath: modelPath)
        let compiledURL = try compile(at: url)
        print("Loading model \(compiledURL)")
        let model = try LanguageModel.loadCompiled(url: compiledURL, computeUnits: computeUnits.asMLComputeUnits)

        // Using greedy generation for now
        var config = model.defaultGenerationConfig
        config.doSample = false
        config.maxNewTokens = maxLength

        print("Warming up...")
        generate(model: model, config: config, prompt: prompt, printOutput: false)

        print("Generating")
        generate(model: model, config: config, prompt: prompt)
    }
}

@available(iOS 16.2, macOS 13.1, *)
enum ComputeUnits: String, ExpressibleByArgument, CaseIterable {
    case all, cpuAndGPU, cpuOnly, cpuAndNeuralEngine
    var asMLComputeUnits: MLComputeUnits {
        switch self {
        case .all: .all
        case .cpuAndGPU: .cpuAndGPU
        case .cpuOnly: .cpuOnly
        case .cpuAndNeuralEngine: .cpuAndNeuralEngine
        }
    }
}

if #available(iOS 16.2, macOS 13.1, *) {
    TransformersCLI.main()
} else {
    print("Unsupported OS")
}

extension Double {
    func formatted(_ format: String) -> String {
        String(format: "\(format)", self)
    }
}

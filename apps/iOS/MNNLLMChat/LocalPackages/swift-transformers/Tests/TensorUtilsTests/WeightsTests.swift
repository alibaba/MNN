@testable import Hub
@testable import TensorUtils
import XCTest

class WeightsTests: XCTestCase {
    let downloadDestination: URL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!.appending(component: "huggingface-tests")

    var hubApi: HubApi { HubApi(downloadBase: downloadDestination) }

    func testLoadWeightsFromFileURL() async throws {
        let repo = "google/bert_uncased_L-2_H-128_A-2"
        let modelDir = try await hubApi.snapshot(from: repo, matching: ["config.json", "model.safetensors"])

        let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: [.isReadableKey])
        XCTAssertTrue(files.contains(where: { $0.lastPathComponent == "config.json" }))
        XCTAssertTrue(files.contains(where: { $0.lastPathComponent == "model.safetensors" }))

        let modelFile = modelDir.appending(path: "/model.safetensors")
        let weights = try Weights.from(fileURL: modelFile)
        XCTAssertEqual(weights["bert.embeddings.LayerNorm.bias"]!.dataType, .float32)
        XCTAssertEqual(weights["bert.embeddings.LayerNorm.bias"]!.count, 128)
        XCTAssertEqual(weights["bert.embeddings.LayerNorm.bias"]!.shape.count, 1)

        XCTAssertEqual(weights["bert.embeddings.word_embeddings.weight"]!.dataType, .float32)
        XCTAssertEqual(weights["bert.embeddings.word_embeddings.weight"]!.count, 3906816)
        XCTAssertEqual(weights["bert.embeddings.word_embeddings.weight"]!.shape.count, 2)

        XCTAssertEqual(weights["bert.embeddings.word_embeddings.weight"]![[0, 0]].floatValue, -0.0041, accuracy: 1e-3)
        XCTAssertEqual(weights["bert.embeddings.word_embeddings.weight"]![[3, 4]].floatValue, 0.0037, accuracy: 1e-3)
        XCTAssertEqual(weights["bert.embeddings.word_embeddings.weight"]![[5, 3]].floatValue, -0.5371, accuracy: 1e-3)
        XCTAssertEqual(weights["bert.embeddings.word_embeddings.weight"]![[7, 8]].floatValue, 0.0460, accuracy: 1e-3)
        XCTAssertEqual(weights["bert.embeddings.word_embeddings.weight"]![[11, 7]].floatValue, -0.0058, accuracy: 1e-3)
    }

    func testSafetensorReadTensor1D() throws {
        let modelFile = Bundle.module.url(forResource: "tensor-1d-int32", withExtension: "safetensors")!
        let weights: Weights = try Weights.from(fileURL: modelFile)
        let tensor = weights["embedding"]!
        XCTAssertEqual(tensor.dataType, .int32)
        XCTAssertEqual(tensor[[0]], 1)
        XCTAssertEqual(tensor[[1]], 2)
        XCTAssertEqual(tensor[[2]], 3)
    }

    func testSafetensorReadTensor2D() throws {
        let modelFile = Bundle.module.url(forResource: "tensor-2d-float64", withExtension: "safetensors")!
        let weights: Weights = try Weights.from(fileURL: modelFile)
        let tensor = weights["embedding"]!
        XCTAssertEqual(tensor.dataType, .float64)
        XCTAssertEqual(tensor[[0, 0]], 1)
        XCTAssertEqual(tensor[[0, 1]], 2)
        XCTAssertEqual(tensor[[0, 2]], 3)
        XCTAssertEqual(tensor[[1, 0]], 24)
        XCTAssertEqual(tensor[[1, 1]], 25)
        XCTAssertEqual(tensor[[1, 2]], 26)
    }

    func testSafetensorReadTensor3D() throws {
        let modelFile = Bundle.module.url(forResource: "tensor-3d-float32", withExtension: "safetensors")!
        let weights: Weights = try Weights.from(fileURL: modelFile)
        let tensor = weights["embedding"]!
        XCTAssertEqual(tensor.dataType, .float32)
        XCTAssertEqual(tensor[[0, 0, 0]], 22)
        XCTAssertEqual(tensor[[0, 0, 1]], 23)
        XCTAssertEqual(tensor[[0, 0, 2]], 24)
        XCTAssertEqual(tensor[[0, 1, 0]], 11)
        XCTAssertEqual(tensor[[0, 1, 1]], 12)
        XCTAssertEqual(tensor[[0, 1, 2]], 13)
        XCTAssertEqual(tensor[[1, 0, 0]], 2)
        XCTAssertEqual(tensor[[1, 0, 1]], 3)
        XCTAssertEqual(tensor[[1, 0, 2]], 4)
        XCTAssertEqual(tensor[[1, 1, 0]], 1)
        XCTAssertEqual(tensor[[1, 1, 1]], 2)
        XCTAssertEqual(tensor[[1, 1, 2]], 3)
    }

    func testSafetensorReadTensor4D() throws {
        let modelFile = Bundle.module.url(forResource: "tensor-4d-float32", withExtension: "safetensors")!
        let weights: Weights = try Weights.from(fileURL: modelFile)
        let tensor = weights["embedding"]!
        XCTAssertEqual(tensor.dataType, .float32)
        XCTAssertEqual(tensor[[0, 0, 0, 0]], 11)
        XCTAssertEqual(tensor[[0, 0, 0, 1]], 12)
        XCTAssertEqual(tensor[[0, 0, 0, 2]], 13)
        XCTAssertEqual(tensor[[0, 0, 1, 0]], 1)
        XCTAssertEqual(tensor[[0, 0, 1, 1]], 2)
        XCTAssertEqual(tensor[[0, 0, 1, 2]], 3)
        XCTAssertEqual(tensor[[0, 0, 2, 0]], 4)
        XCTAssertEqual(tensor[[0, 0, 2, 1]], 5)
        XCTAssertEqual(tensor[[0, 0, 2, 2]], 6)
        XCTAssertEqual(tensor[[1, 0, 0, 0]], 22)
        XCTAssertEqual(tensor[[1, 0, 0, 1]], 23)
        XCTAssertEqual(tensor[[1, 0, 0, 2]], 24)
        XCTAssertEqual(tensor[[1, 0, 1, 0]], 15)
        XCTAssertEqual(tensor[[1, 0, 1, 1]], 16)
        XCTAssertEqual(tensor[[1, 0, 1, 2]], 17)
        XCTAssertEqual(tensor[[1, 0, 2, 0]], 17)
        XCTAssertEqual(tensor[[1, 0, 2, 1]], 18)
        XCTAssertEqual(tensor[[1, 0, 2, 2]], 19)
    }
}

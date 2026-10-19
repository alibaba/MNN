//
//  HubApiTests.swift
//
//  Created by Pedro Cuenca on 20231230.
//

@testable import Hub
import XCTest

class HubApiTests: XCTestCase {
    override func setUp() {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    // MARK: use a specific revision for these tests

    func testFilenameRetrieval() async {
        do {
            let filenames = try await Hub.getFilenames(from: "coreml-projects/Llama-2-7b-chat-coreml")
            XCTAssertEqual(filenames.count, 13)
        } catch {
            XCTFail("\(error)")
        }
    }

    func testFilenameRetrievalWithGlob() async {
        do {
            try await {
                let filenames = try await Hub.getFilenames(from: "coreml-projects/Llama-2-7b-chat-coreml", matching: "*.json")
                XCTAssertEqual(
                    Set(filenames),
                    Set([
                        "config.json", "tokenizer.json", "tokenizer_config.json",
                        "llama-2-7b-chat.mlpackage/Manifest.json",
                        "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
                        "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
                    ])
                )
            }()

            // Glob patterns are case sensitive
            try await {
                let filenames = try await Hub.getFilenames(from: "coreml-projects/Llama-2-7b-chat-coreml", matching: "*.JSON")
                XCTAssertEqual(
                    filenames,
                    []
                )
            }()
        } catch {
            XCTFail("\(error)")
        }
    }

    func testFilenameRetrievalFromDirectories() async {
        do {
            // Contents of all directories matching a pattern
            let filenames = try await Hub.getFilenames(from: "coreml-projects/Llama-2-7b-chat-coreml", matching: "*.mlpackage/*")
            XCTAssertEqual(
                Set(filenames),
                Set([
                    "llama-2-7b-chat.mlpackage/Manifest.json",
                    "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
                    "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
                    "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel",
                    "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/weights/weight.bin",

                ])
            )
        } catch {
            XCTFail("\(error)")
        }
    }

    func testFilenameRetrievalWithMultiplePatterns() async {
        do {
            let patterns = ["config.json", "tokenizer.json", "tokenizer_*.json"]
            let filenames = try await Hub.getFilenames(from: "coreml-projects/Llama-2-7b-chat-coreml", matching: patterns)
            XCTAssertEqual(
                Set(filenames),
                Set(["config.json", "tokenizer.json", "tokenizer_config.json"])
            )
        } catch {
            XCTFail("\(error)")
        }
    }

    func testGetFileMetadata() async throws {
        do {
            let url = URL(string: "https://huggingface.co/enterprise-explorers/Llama-2-7b-chat-coreml/resolve/main/config.json")
            let metadata = try await Hub.getFileMetadata(fileURL: url!)

            XCTAssertNotNil(metadata.commitHash)
            XCTAssertNotNil(metadata.etag)
            XCTAssertEqual(URL(string: metadata.location)?.path, "/api/resolve-cache/models\(url!.path.replacingOccurrences(of: "resolve/main", with: metadata.commitHash!))")
            XCTAssertEqual(metadata.size, 163)
        } catch {
            XCTFail("\(error)")
        }
    }

    func testGetFileMetadataBlobPath() async throws {
        do {
            let url = URL(string: "https://huggingface.co/enterprise-explorers/Llama-2-7b-chat-coreml/resolve/main/config.json")
            let metadata = try await Hub.getFileMetadata(fileURL: url!)

            XCTAssertNotNil(metadata.commitHash)
            XCTAssertTrue(metadata.etag != nil && metadata.etag!.hasPrefix("d6ceb9"))
            XCTAssertEqual(URL(string: metadata.location)?.path, "/api/resolve-cache/models\(url!.path.replacingOccurrences(of: "resolve/main", with: metadata.commitHash!))")
            XCTAssertEqual(metadata.size, 163)
        } catch {
            XCTFail("\(error)")
        }
    }

    func testGetFileMetadataWithRevision() async throws {
        do {
            let revision = "f2c752cfc5c0ab6f4bdec59acea69eefbee381c2"
            let url = URL(string: "https://huggingface.co/julien-c/dummy-unknown/resolve/\(revision)/config.json")
            let metadata = try await Hub.getFileMetadata(fileURL: url!)

            XCTAssertEqual(metadata.commitHash, revision)
            XCTAssertNotNil(metadata.etag)
            XCTAssertGreaterThan(metadata.etag!.count, 0)
            XCTAssertEqual(URL(string: metadata.location)?.path, "/api/resolve-cache/models\(url!.path.replacingOccurrences(of: "resolve/\(revision)", with: metadata.commitHash!))")
            XCTAssertEqual(metadata.size, 851)
        } catch {
            XCTFail("\(error)")
        }
    }

    func testGetFileMetadataWithBlobSearch() async throws {
        let repo = "coreml-projects/Llama-2-7b-chat-coreml"
        let metadataFromBlob = try await Hub.getFileMetadata(from: repo, matching: "*.json")
        for metadata in metadataFromBlob {
            XCTAssertNotNil(metadata.commitHash)
            XCTAssertNotNil(metadata.etag)
            XCTAssertGreaterThan(metadata.etag!.count, 0)
            XCTAssertGreaterThan(metadata.size!, 0)
        }
    }

    /// Verify with `curl -I https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel`
    func testGetLargeFileMetadata() async throws {
        do {
            let revision = "eaf97358a37d03fd48e5a87d15aff2e8423c1afb"
            let etag = "fc329090bfbb2570382c9af997cffd5f4b78b39b8aeca62076db69534e020107"
            let location = "https://cdn-lfs.hf.co/repos/4a/4e/4a4e587f66a2979dcd75e1d7324df8ee9ef74be3582a05bea31c2c26d0d467d0/fc329090bfbb2570382c9af997cffd5f4b78b39b8aeca62076db69534e020107?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27model.mlmodel%3B+filename%3D%22model.mlmodel"
            let size = 504766

            let url = URL(string: "https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel")
            let metadata = try await Hub.getFileMetadata(fileURL: url!)

            XCTAssertEqual(metadata.commitHash, revision)
            XCTAssertNotNil(metadata.etag, etag)
            XCTAssertTrue(metadata.location.contains(location))
            XCTAssertEqual(metadata.size, size)
        } catch {
            XCTFail("\(error)")
        }
    }
}

class SnapshotDownloadTests: XCTestCase {
    let repo = "coreml-projects/Llama-2-7b-chat-coreml"
    let lfsRepo = "pcuenq/smol-lfs"
    let downloadDestination: URL = {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return base.appending(component: "huggingface-tests")
    }()

    override func setUp() { }

    override func tearDown() {
        do {
            try FileManager.default.removeItem(at: downloadDestination)
        } catch {
            print("Can't remove test download destination \(downloadDestination), error: \(error)")
        }
    }

    func getRelativeFiles(url: URL, repo: String) -> [String] {
        var filenames: [String] = []
        let prefix = downloadDestination.appending(path: "models/\(repo)").path.appending("/")

        if let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles], errorHandler: nil) {
            for case let fileURL as URL in enumerator {
                do {
                    let resourceValues = try fileURL.resourceValues(forKeys: [.isRegularFileKey])
                    if resourceValues.isRegularFile == true {
                        filenames.append(String(fileURL.path.suffix(from: prefix.endIndex)))
                    }
                } catch {
                    print("Error reading file resources: \(error)")
                }
            }
        }
        return filenames
    }

    func testDownload() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        // Add debug prints
        print("Download destination before: \(downloadDestination.path)")

        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        // Add more debug prints
        print("Downloaded to: \(downloadedTo.path)")

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        print("Downloaded filenames: \(downloadedFilenames)")
        print("Prefix used in getRelativeFiles: \(downloadDestination.appending(path: "models/\(repo)").path)")

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 6)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        XCTAssertEqual(
            Set(downloadedFilenames),
            Set([
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "llama-2-7b-chat.mlpackage/Manifest.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
            ])
        )
    }

    /// Background sessions get rate limited by the OS, see discussion here: https://github.com/huggingface/swift-transformers/issues/61
    /// Test only one file at a time
    func testDownloadInBackground() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination, useBackgroundSession: true)
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }
        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 1)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        XCTAssertEqual(
            Set(downloadedFilenames),
            Set([
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
            ])
        )
    }

    func testCustomEndpointDownload() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination, endpoint: "https://hf-mirror.com")
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }
        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 6)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        XCTAssertEqual(
            Set(downloadedFilenames),
            Set([
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "llama-2-7b-chat.mlpackage/Manifest.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
            ])
        )
    }

    func testDownloadFileMetadata() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }
        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 6)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        XCTAssertEqual(
            Set(downloadedFilenames),
            Set([
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "llama-2-7b-chat.mlpackage/Manifest.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
            ])
        )

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")
        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: repo)
        XCTAssertEqual(
            Set(downloadedMetadataFilenames),
            Set([
                ".cache/huggingface/download/config.json.metadata",
                ".cache/huggingface/download/tokenizer.json.metadata",
                ".cache/huggingface/download/tokenizer_config.json.metadata",
                ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Manifest.json.metadata",
                ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json.metadata",
                ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json.metadata",
            ])
        )
    }

    func testDownloadFileMetadataExists() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 6)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        XCTAssertEqual(
            Set(downloadedFilenames),
            Set([
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "llama-2-7b-chat.mlpackage/Manifest.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
            ])
        )

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let configPath = downloadedTo.appending(path: "config.json")
        var attributes = try FileManager.default.attributesOfItem(atPath: configPath.path)
        let originalTimestamp = attributes[.modificationDate] as! Date

        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: repo)
        XCTAssertEqual(
            Set(downloadedMetadataFilenames),
            Set([
                ".cache/huggingface/download/config.json.metadata",
                ".cache/huggingface/download/tokenizer.json.metadata",
                ".cache/huggingface/download/tokenizer_config.json.metadata",
                ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Manifest.json.metadata",
                ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json.metadata",
                ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json.metadata",
            ])
        )

        let _ = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        attributes = try FileManager.default.attributesOfItem(atPath: configPath.path)
        let secondDownloadTimestamp = attributes[.modificationDate] as! Date

        // File will not be downloaded again thus last modified date will remain unchanged
        XCTAssertTrue(originalTimestamp == secondDownloadTimestamp)
    }

    func testDownloadFileMetadataSame() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "tokenizer.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 1)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        XCTAssertEqual(Set(downloadedFilenames), Set(["tokenizer.json"]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let metadataPath = metadataDestination.appending(path: "tokenizer.json.metadata")

        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: repo)
        XCTAssertEqual(
            Set(downloadedMetadataFilenames),
            Set([
                ".cache/huggingface/download/tokenizer.json.metadata",
            ])
        )

        let originalMetadata = try String(contentsOf: metadataPath, encoding: .utf8)

        let _ = try await hubApi.snapshot(from: repo, matching: "tokenizer.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        let secondDownloadMetadata = try String(contentsOf: metadataPath, encoding: .utf8)

        // File hasn't changed so commit hash and etag will be identical
        let originalArr = originalMetadata.components(separatedBy: .newlines)
        let secondDownloadArr = secondDownloadMetadata.components(separatedBy: .newlines)

        XCTAssertTrue(originalArr[0] == secondDownloadArr[0])
        XCTAssertTrue(originalArr[1] == secondDownloadArr[1])
    }

    func testDownloadFileMetadataCorrupted() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 6)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        XCTAssertEqual(
            Set(downloadedFilenames),
            Set([
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "llama-2-7b-chat.mlpackage/Manifest.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
            ])
        )

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let configPath = downloadedTo.appending(path: "config.json")
        var attributes = try FileManager.default.attributesOfItem(atPath: configPath.path)
        let originalTimestamp = attributes[.modificationDate] as! Date

        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: repo)
        XCTAssertEqual(
            Set(downloadedMetadataFilenames),
            Set([
                ".cache/huggingface/download/config.json.metadata",
                ".cache/huggingface/download/tokenizer.json.metadata",
                ".cache/huggingface/download/tokenizer_config.json.metadata",
                ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Manifest.json.metadata",
                ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json.metadata",
                ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json.metadata",
            ])
        )

        // Corrupt config.json.metadata
        print("Testing corrupted file.")
        try "a".write(to: metadataDestination.appendingPathComponent("config.json.metadata"), atomically: true, encoding: .utf8)

        let _ = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        attributes = try FileManager.default.attributesOfItem(atPath: configPath.path)
        let secondDownloadTimestamp = attributes[.modificationDate] as! Date

        // File will be downloaded again thus last modified date will change
        XCTAssertTrue(originalTimestamp != secondDownloadTimestamp)

        // Corrupt config.metadata again
        print("Testing corrupted timestamp.")
        try "a\nb\nc\n".write(to: metadataDestination.appendingPathComponent("config.json.metadata"), atomically: true, encoding: .utf8)

        let _ = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        attributes = try FileManager.default.attributesOfItem(atPath: configPath.path)
        let thirdDownloadTimestamp = attributes[.modificationDate] as! Date

        // File will be downloaded again thus last modified date will change
        XCTAssertTrue(originalTimestamp != thirdDownloadTimestamp)
    }

    func testDownloadLargeFileMetadataCorrupted() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.mlmodel") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 1)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        XCTAssertEqual(
            Set(downloadedFilenames),
            Set(["llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel"])
        )

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let modelPath = downloadedTo.appending(path: "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel")
        var attributes = try FileManager.default.attributesOfItem(atPath: modelPath.path)
        let originalTimestamp = attributes[.modificationDate] as! Date

        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: repo)
        XCTAssertEqual(
            Set(downloadedMetadataFilenames),
            Set([
                ".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel.metadata",
            ])
        )

        // Corrupt model.metadata etag
        print("Testing corrupted etag.")
        let corruptedMetadataString = "a\nfc329090bfbb2570382c9af997cffd5f4b78b39b8aeca62076db69534e020108\n0\n"
        let metadataFile = metadataDestination.appendingPathComponent("llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel.metadata")
        try corruptedMetadataString.write(to: metadataFile, atomically: true, encoding: .utf8)

        let _ = try await hubApi.snapshot(from: repo, matching: "*.mlmodel") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        attributes = try FileManager.default.attributesOfItem(atPath: modelPath.path)
        let thirdDownloadTimestamp = attributes[.modificationDate] as! Date

        // File will not be downloaded again because this is an LFS file.
        // While downloading LFS files, we first check if local file ETag is the same as remote ETag.
        // If that's the case we just update the metadata and keep the local file.
        XCTAssertEqual(originalTimestamp, thirdDownloadTimestamp)

        let metadataString = try String(contentsOfFile: metadataFile.path)

        // Updated metadata file needs to have the correct commit hash, etag and timestamp.
        // This is being updated because the local etag (SHA256 checksum) matches the remote etag
        XCTAssertNotEqual(metadataString, corruptedMetadataString)
    }

    func testDownloadLargeFile() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.mlmodel") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 1)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        XCTAssertEqual(Set(downloadedFilenames), Set(["llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel"]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")
        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: repo)
        XCTAssertEqual(
            Set(downloadedMetadataFilenames),
            Set([".cache/huggingface/download/llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel.metadata"])
        )

        let metadataFile = metadataDestination.appendingPathComponent("llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/model.mlmodel.metadata")
        let metadataString = try String(contentsOfFile: metadataFile.path)

        let expected = "eaf97358a37d03fd48e5a87d15aff2e8423c1afb\nfc329090bfbb2570382c9af997cffd5f4b78b39b8aeca62076db69534e020107"
        XCTAssertTrue(metadataString.contains(expected))
    }

    func testDownloadSmolLargeFile() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: lfsRepo, matching: "x.bin") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 1)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(lfsRepo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: lfsRepo)
        XCTAssertEqual(Set(downloadedFilenames), Set(["x.bin"]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")
        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: lfsRepo)
        XCTAssertEqual(
            Set(downloadedMetadataFilenames),
            Set([".cache/huggingface/download/x.bin.metadata"])
        )

        let metadataFile = metadataDestination.appendingPathComponent("x.bin.metadata")
        let metadataString = try String(contentsOfFile: metadataFile.path)

        let expected = "77b984598d90af6143d73d5a2d6214b23eba7e27\n98ea6e4f216f2fb4b69fff9b3a44842c38686ca685f3f55dc48c5d3fb1107be4"
        XCTAssertTrue(metadataString.contains(expected))
    }

    func testRegexValidation() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil
        let downloadedTo = try await hubApi.snapshot(from: lfsRepo, matching: "x.bin") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 1)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(lfsRepo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: lfsRepo)
        XCTAssertEqual(Set(downloadedFilenames), Set(["x.bin"]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")
        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: lfsRepo)
        XCTAssertEqual(
            Set(downloadedMetadataFilenames),
            Set([".cache/huggingface/download/x.bin.metadata"])
        )

        let metadataFile = metadataDestination.appendingPathComponent("x.bin.metadata")
        let metadataString = try String(contentsOfFile: metadataFile.path)
        let metadataArr = metadataString.components(separatedBy: .newlines)

        let commitHash = metadataArr[0]
        let etag = metadataArr[1]

        XCTAssertTrue(hubApi.isValidHash(hash: commitHash, pattern: hubApi.commitHashPattern))
        XCTAssertTrue(hubApi.isValidHash(hash: etag, pattern: hubApi.sha256Pattern))

        XCTAssertFalse(hubApi.isValidHash(hash: "\(commitHash)a", pattern: hubApi.commitHashPattern))
        XCTAssertFalse(hubApi.isValidHash(hash: "\(etag)a", pattern: hubApi.sha256Pattern))
    }

    func testLFSFileNoMetadata() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: lfsRepo, matching: "x.bin") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 1)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(lfsRepo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: lfsRepo)
        XCTAssertEqual(Set(downloadedFilenames), Set(["x.bin"]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let filePath = downloadedTo.appending(path: "x.bin")
        var attributes = try FileManager.default.attributesOfItem(atPath: filePath.path)
        let originalTimestamp = attributes[.modificationDate] as! Date

        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: lfsRepo)
        XCTAssertEqual(
            Set(downloadedMetadataFilenames),
            Set([".cache/huggingface/download/x.bin.metadata"])
        )

        let metadataFile = metadataDestination.appendingPathComponent("x.bin.metadata")
        try FileManager.default.removeItem(atPath: metadataFile.path)

        let _ = try await hubApi.snapshot(from: lfsRepo, matching: "x.bin") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        attributes = try FileManager.default.attributesOfItem(atPath: filePath.path)
        let secondDownloadTimestamp = attributes[.modificationDate] as! Date

        // File will not be downloaded again thus last modified date will remain unchanged
        XCTAssertTrue(originalTimestamp == secondDownloadTimestamp)
        XCTAssertTrue(FileManager.default.fileExists(atPath: metadataDestination.path))

        let metadataString = try String(contentsOfFile: metadataFile.path)
        let expected = "77b984598d90af6143d73d5a2d6214b23eba7e27\n98ea6e4f216f2fb4b69fff9b3a44842c38686ca685f3f55dc48c5d3fb1107be4"

        XCTAssertTrue(metadataString.contains(expected))
    }

    func testLFSFileCorruptedMetadata() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: lfsRepo, matching: "x.bin") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 1)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(lfsRepo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: lfsRepo)
        XCTAssertEqual(Set(downloadedFilenames), Set(["x.bin"]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let filePath = downloadedTo.appending(path: "x.bin")
        var attributes = try FileManager.default.attributesOfItem(atPath: filePath.path)
        let originalTimestamp = attributes[.modificationDate] as! Date

        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: lfsRepo)
        XCTAssertEqual(
            Set(downloadedMetadataFilenames),
            Set([".cache/huggingface/download/x.bin.metadata"])
        )

        let metadataFile = metadataDestination.appendingPathComponent("x.bin.metadata")
        try "a".write(to: metadataFile, atomically: true, encoding: .utf8)

        let _ = try await hubApi.snapshot(from: lfsRepo, matching: "x.bin") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        attributes = try FileManager.default.attributesOfItem(atPath: filePath.path)
        let secondDownloadTimestamp = attributes[.modificationDate] as! Date

        // File will not be downloaded again thus last modified date will remain unchanged
        XCTAssertTrue(originalTimestamp == secondDownloadTimestamp)
        XCTAssertTrue(FileManager.default.fileExists(atPath: metadataDestination.path))

        let metadataString = try String(contentsOfFile: metadataFile.path)
        let expected = "77b984598d90af6143d73d5a2d6214b23eba7e27\n98ea6e4f216f2fb4b69fff9b3a44842c38686ca685f3f55dc48c5d3fb1107be4"

        XCTAssertTrue(metadataString.contains(expected))
    }

    func testNonLFSFileRedownload() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: repo, matching: "config.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 1)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        XCTAssertEqual(Set(downloadedFilenames), Set(["config.json"]))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let filePath = downloadedTo.appending(path: "config.json")
        var attributes = try FileManager.default.attributesOfItem(atPath: filePath.path)
        let originalTimestamp = attributes[.modificationDate] as! Date

        let downloadedMetadataFilenames = getRelativeFiles(url: metadataDestination, repo: repo)
        XCTAssertEqual(
            Set(downloadedMetadataFilenames),
            Set([".cache/huggingface/download/config.json.metadata"])
        )

        let metadataFile = metadataDestination.appendingPathComponent("config.json.metadata")
        try FileManager.default.removeItem(atPath: metadataFile.path)

        let _ = try await hubApi.snapshot(from: repo, matching: "config.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        attributes = try FileManager.default.attributesOfItem(atPath: filePath.path)
        let secondDownloadTimestamp = attributes[.modificationDate] as! Date

        // File will be downloaded again thus last modified date will change
        XCTAssertTrue(originalTimestamp != secondDownloadTimestamp)
        XCTAssertTrue(FileManager.default.fileExists(atPath: metadataDestination.path))

        let metadataString = try String(contentsOfFile: metadataFile.path)
        let expected = "eaf97358a37d03fd48e5a87d15aff2e8423c1afb\nd6ceb92ce9e3c83ab146dc8e92a93517ac1cc66f"

        XCTAssertTrue(metadataString.contains(expected))
    }

    func testOfflineModeReturnsDestination() async throws {
        var hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        var downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")

            lastProgress = progress
        }

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 6)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        hubApi = HubApi(downloadBase: downloadDestination, useOfflineMode: true)

        downloadedTo = try await hubApi.snapshot(from: repo, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 6)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))
    }

    func testOfflineModeThrowsError() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination, useOfflineMode: true)

        do {
            try await hubApi.snapshot(from: repo, matching: "*.json")
            XCTFail("Expected an error to be thrown")
        } catch let error as HubApi.EnvironmentError {
            switch error {
            case let .offlineModeError(message):
                XCTAssertEqual(message, "Repository not available locally")
            default:
                XCTFail("Wrong error type: \(error)")
            }
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testOfflineModeWithoutMetadata() async throws {
        var hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: lfsRepo, matching: "*") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")

            lastProgress = progress
        }

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 2)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(lfsRepo)"))

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let metadataFile = metadataDestination.appendingPathComponent("x.bin.metadata")
        try FileManager.default.removeItem(atPath: metadataFile.path)

        hubApi = HubApi(downloadBase: downloadDestination, useOfflineMode: true)

        do {
            try await hubApi.snapshot(from: lfsRepo, matching: "*")
            XCTFail("Expected an error to be thrown")
        } catch let error as HubApi.EnvironmentError {
            switch error {
            case let .offlineModeError(message):
                XCTAssertEqual(message, "Metadata not available for x.bin")
            default:
                XCTFail("Wrong error type: \(error)")
            }
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testOfflineModeWithCorruptedLFSMetadata() async throws {
        var hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: lfsRepo, matching: "*") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")

            lastProgress = progress
        }

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 2)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(lfsRepo)"))

        let metadataDestination = downloadedTo.appendingPathComponent(".cache/huggingface/download").appendingPathComponent("x.bin.metadata")

        try "77b984598d90af6143d73d5a2d6214b23eba7e27\n98ea6e4f216f2ab4b69fff9b3a44842c38686ca685f3f55dc48c5d3fb1107be4\n0\n".write(to: metadataDestination, atomically: true, encoding: .utf8)

        hubApi = HubApi(downloadBase: downloadDestination, useOfflineMode: true)

        do {
            try await hubApi.snapshot(from: lfsRepo, matching: "*")
            XCTFail("Expected an error to be thrown")
        } catch let error as HubApi.EnvironmentError {
            switch error {
            case let .fileIntegrityError(message):
                XCTAssertEqual(message, "Hash mismatch for x.bin")
            default:
                XCTFail("Wrong error type: \(error)")
            }
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testOfflineModeWithNoFiles() async throws {
        var hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let downloadedTo = try await hubApi.snapshot(from: lfsRepo, matching: "x.bin") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")

            lastProgress = progress
        }

        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 1)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(lfsRepo)"))

        let fileDestination = downloadedTo.appendingPathComponent("x.bin")
        try FileManager.default.removeItem(at: fileDestination)

        hubApi = HubApi(downloadBase: downloadDestination, useOfflineMode: true)

        do {
            try await hubApi.snapshot(from: lfsRepo, matching: "x.bin")
            XCTFail("Expected an error to be thrown")
        } catch let error as HubApi.EnvironmentError {
            switch error {
            case let .offlineModeError(message):
                XCTAssertEqual(message, "No files available locally for this repository")
            default:
                XCTFail("Wrong error type: \(error)")
            }
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testResumeDownloadFromEmptyIncomplete() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil
        var downloadedTo = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Caches/huggingface-tests/models/coreml-projects/Llama-2-7b-chat-coreml")

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let url = URL(string: "https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/config.json")!
        let etag = try await Hub.getFileMetadata(fileURL: url).etag!

        try FileManager.default.createDirectory(at: metadataDestination, withIntermediateDirectories: true, attributes: nil)
        try "".write(to: metadataDestination.appendingPathComponent("config.json.\(etag).incomplete"), atomically: true, encoding: .utf8)
        downloadedTo = try await hubApi.snapshot(from: repo, matching: "config.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }
        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 1)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        let fileContents = try String(contentsOfFile: downloadedTo.appendingPathComponent("config.json").path)

        let expected = """
        {
          "architectures": [
            "LlamaForCausalLM"
          ],
          "bos_token_id": 1,
          "eos_token_id": 2,
          "model_type": "llama",
          "pad_token_id": 0,
          "vocab_size": 32000
        }
        """
        XCTAssertTrue(fileContents.contains(expected))
    }

    func testResumeDownloadFromNonEmptyIncomplete() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil
        var downloadedTo = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/huggingface-tests/models/coreml-projects/Llama-2-7b-chat-coreml")

        let metadataDestination = downloadedTo.appending(component: ".cache/huggingface/download")

        let url = URL(string: "https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/config.json")!
        let etag = try await Hub.getFileMetadata(fileURL: url).etag!

        try FileManager.default.createDirectory(at: metadataDestination, withIntermediateDirectories: true, attributes: nil)
        try "X".write(to: metadataDestination.appendingPathComponent("config.json.\(etag).incomplete"), atomically: true, encoding: .utf8)
        downloadedTo = try await hubApi.snapshot(from: repo, matching: "config.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }
        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 1)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))

        let fileContents = try String(contentsOfFile: downloadedTo.appendingPathComponent("config.json").path)

        let expected = """
        X
          "architectures": [
            "LlamaForCausalLM"
          ],
          "bos_token_id": 1,
          "eos_token_id": 2,
          "model_type": "llama",
          "pad_token_id": 0,
          "vocab_size": 32000
        }
        """
        XCTAssertTrue(fileContents.contains(expected))
    }

    func testRealDownloadInterruptionAndResumption() async throws {
        // Use the DepthPro model weights file
        let targetFile = "SAM 2 Studio 1.1.zip"
        let repo = "coreml-projects/sam-2-studio"
        let hubApi = HubApi(downloadBase: downloadDestination)

        // Create expectation for first progress update
        let progressExpectation = expectation(description: "First progress update received")

        // Create a task for the download
        let downloadTask = Task {
            try await hubApi.snapshot(from: repo, matching: targetFile) { progress in
                print("Progress reached 1 \(progress.fractionCompleted * 100)%")
                if progress.fractionCompleted > 0 {
                    progressExpectation.fulfill()
                }
            }
        }

        // Wait for the first progress update
        await fulfillment(of: [progressExpectation], timeout: 30.0)

        // Cancel the download once we've seen progress
        downloadTask.cancel()
        try await Task.sleep(nanoseconds: 5_000_000_000)

        // Resume download with a new task
        let downloadedTo = try await hubApi.snapshot(from: repo, matching: targetFile) { progress in
            print("Progress reached 2 \(progress.fractionCompleted * 100)%")
        }

        let filePath = downloadedTo.appendingPathComponent(targetFile)
        XCTAssertTrue(FileManager.default.fileExists(atPath: filePath.path),
                      "Downloaded file should exist at \(filePath.path)")
    }

    func testDownloadWithRevision() async throws {
        let hubApi = HubApi(downloadBase: downloadDestination)
        var lastProgress: Progress? = nil

        let commitHash = "eaf97358a37d03fd48e5a87d15aff2e8423c1afb"
        let downloadedTo = try await hubApi.snapshot(from: repo, revision: commitHash, matching: "*.json") { progress in
            print("Total Progress: \(progress.fractionCompleted)")
            print("Files Completed: \(progress.completedUnitCount) of \(progress.totalUnitCount)")
            lastProgress = progress
        }

        let downloadedFilenames = getRelativeFiles(url: downloadDestination, repo: repo)
        XCTAssertEqual(lastProgress?.fractionCompleted, 1)
        XCTAssertEqual(lastProgress?.completedUnitCount, 6)
        XCTAssertEqual(downloadedTo, downloadDestination.appending(path: "models/\(repo)"))
        XCTAssertEqual(
            Set(downloadedFilenames),
            Set([
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "llama-2-7b-chat.mlpackage/Manifest.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/FeatureDescriptions.json",
                "llama-2-7b-chat.mlpackage/Data/com.apple.CoreML/Metadata.json",
            ])
        )

        do {
            let revision = "nonexistent-revision"
            try await hubApi.snapshot(from: repo, revision: revision, matching: "*.json")
            XCTFail("Expected an error to be thrown")
        } catch let error as Hub.HubClientError {
            switch error {
            case .fileNotFound:
                break // Error type is correct
            default:
                XCTFail("Wrong error type: \(error)")
            }
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
}

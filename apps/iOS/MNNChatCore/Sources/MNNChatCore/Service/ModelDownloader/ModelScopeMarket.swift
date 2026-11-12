import Foundation

/// Only metadata is cached. The repository's last_modified timestamp invalidates its format check.
struct ModelScopeMarket {
    struct Repository: Codable {
        let id: String
        let last_modified: String
        let file_size: Int64?
        let tasks: [String]?
        let tags: [String]?
        let gated: Bool?
        var name: String { String(id.split(separator: "/").last ?? "") }

        var updatedAt: Date? {
            (try? Date(last_modified, strategy: .iso8601)) ??
                (try? Date(last_modified, strategy: .iso8601.year().month().day().time(includingFractionalSeconds: true)))
        }

        var eligible: Bool {
            guard id == "MNN/" + name, !name.isEmpty, name != ".", name != "..", gated != true else { return false }
            // Deliberately generous. Smaller but demanding models retain the existing device-fit warning.
            let suffix = name.lowercased()
            return (file_size ?? 0) <= 16 * 1024 * 1024 * 1024 &&
                !suffix.hasSuffix("-npu") && !suffix.hasSuffix("-opencl")
        }
    }

    struct FormatCheck: Codable {
        let revision: String
        let supported: Bool
    }

    /// Match the packages loaded by the chat, Sana and GPU diffusion entry points.
    static func supports(_ name: String, files: Set<String>) -> Bool {
        if name.lowercased().contains("sana") {
            return ["llm", "connector.mnn", "vae_encoder.mnn", "transformer.mnn"].allSatisfy(files.contains)
        }
        if name.lowercased().contains("stable-diffusion") {
            return name.lowercased().contains("gpu") &&
                ["text_encoder.mnn", "unet.mnn", "vae_decoder.mnn"].allSatisfy(files.contains)
        }
        return files.contains("config.json") && files.contains("llm.mnn")
    }

    private struct Page: Decodable {
        struct Contents: Decodable { let models: [Repository]; let total_count: Int; let page_number: Int }
        let data: Contents
    }
    private struct FileList: Decodable {
        struct Contents: Decodable {
            struct Entry: Decodable { let Name: String }
            let Files: [Entry]
        }
        let Data: Contents
    }

    let session: URLSession
    let seedURL: URL?
    let cacheDirectory: URL

    private func request<T: Decodable>(_ url: URL, as: T.Type) async throws -> T {
        let (bytes, response) = try await session.data(for: URLRequest(url: url, timeoutInterval: 15))
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
        return try JSONDecoder().decode(T.self, from: bytes)
    }

    private func cached<T: Decodable>(_ url: URL?, as: T.Type) -> T? {
        guard let url, let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(T.self, from: data)
    }

    private func save<T: Encodable>(_ value: T, to url: URL) {
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        try? JSONEncoder().encode(value).write(to: url, options: .atomic)
    }

    func models() async throws -> [Repository] {
        let snapshot = cacheDirectory.appendingPathComponent("catalog-v1.json")
        do {
            var repositories: [String: Repository] = [:]
            var number = 1
            while true {
                var url = URLComponents(string: "https://modelscope.cn/openapi/v1/models")!
                url.queryItems = [.init(name: "owner", value: "MNN"), .init(name: "page_size", value: "50"),
                                  .init(name: "page_number", value: String(number))]
                let page = try await request(url.url!, as: Page.self).data
                guard page.page_number == number, !page.models.isEmpty || page.total_count == 0 else {
                    throw URLError(.cannotParseResponse)
                }
                for repo in page.models { repositories[repo.id] = repo }
                if number * 50 >= page.total_count { break }
                number += 1
            }

            let checksURL = cacheDirectory.appendingPathComponent("formats-v1.json")
            var checks = cached(seedURL, as: [String: FormatCheck].self) ?? [:]
            checks.merge(cached(checksURL, as: [String: FormatCheck].self) ?? [:]) { _, new in new }
            let candidates = repositories.values.filter(\.eligible)
            var unknown = candidates.filter { checks[$0.id]?.revision != $0.last_modified }.makeIterator()
            await withTaskGroup(of: (String, FormatCheck?).self) { group in
                func enqueue(_ repo: Repository) {
                    group.addTask {
                        var url = URLComponents()
                        url.scheme = "https"
                        url.host = "modelscope.cn"
                        url.path = "/api/v1/models/\(repo.id)/repo/files"
                        url.queryItems = [.init(name: "Root", value: ""), .init(name: "Revision", value: "master")]
                        guard let files = try? await request(url.url!, as: FileList.self).Data.Files else {
                            return (repo.id, nil)
                        }
                        let supported = Self.supports(repo.name, files: Set(files.map(\.Name)))
                        return (repo.id, FormatCheck(revision: repo.last_modified, supported: supported))
                    }
                }
                for _ in 0..<4 { if let repo = unknown.next() { enqueue(repo) } }
                while let (id, check) = await group.next() {
                    if let check { checks[id] = check }
                    if let repo = unknown.next() { enqueue(repo) }
                }
            }
            try Task.checkCancellation()
            // A failed file-list request leaves that revision unverified and eligible for retry.
            let result = candidates.filter { repo in
                guard let check = checks[repo.id] else { return false }
                return check.revision == repo.last_modified && check.supported
            }.sorted { $0.id < $1.id }
            save(checks.filter { repositories[$0.key] != nil }, to: checksURL)
            if candidates.allSatisfy({ checks[$0.id]?.revision == $0.last_modified }) {
                save(result, to: snapshot)
            }
            return result
        } catch {
            try Task.checkCancellation()
            if let previous = cached(snapshot, as: [Repository].self) { return previous }
            throw error
        }
    }
}

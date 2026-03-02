//
//  ModelClient.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/2/20.
//

import Foundation

public enum ModelScopeError: Error {
    case invalidURL
    case invalidResponse
    case downloadCancelled
    case downloadFailed(Error)
    case fileSystemError(Error)
    case invalidData
}

public struct ModelFile: Codable, Sendable {
    public let type: String
    public let name: String
    public let path: String
    public let size: Int
    public let revision: String
    
    enum CodingKeys: String, CodingKey {
        case type = "Type"
        case name = "Name"
        case path = "Path"
        case size = "Size"
        case revision = "Revision"
    }
}

struct ModelResponse: Codable, Sendable {
    let code: Int
    let data: ModelData
    
    enum CodingKeys: String, CodingKey {
        case code = "Code"
        case data = "Data"
    }
}

struct ModelData: Codable, Sendable {
    let files: [ModelFile]
    
    enum CodingKeys: String, CodingKey {
        case files = "Files"
    }
}

struct FileStatus: Codable {
    let path: String
    let size: Int
    let revision: String
    let lastModified: Date
}

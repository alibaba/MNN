//
//  ModelClient.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/3.
//

import Hub
import Foundation

struct ModelInfo: Codable {
    let modelId: String
    let createdAt: String
    let downloads: Int
    let tags: [String]
    
    var name: String {
        modelId.removingTaobaoPrefix()
    }
    
    var isDownloaded: Bool = false
    
    var localPath: String {
        return HubApi.shared.localRepoLocation(HubApi.Repo.init(id: modelId)).path
    }
    
    private enum CodingKeys: String, CodingKey {
        case modelId
        case tags
        case downloads
        case createdAt
    }
}

struct RepoInfo: Codable {
    let modelId: String
    let sha: String
    let siblings: [Sibling]

    struct Sibling: Codable {
        let rfilename: String
    }
}

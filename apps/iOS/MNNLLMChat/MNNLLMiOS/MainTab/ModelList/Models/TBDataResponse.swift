//
//  TBDataResponse.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/7/9.
//

import Foundation

struct TBDataResponse: Codable {
    let tagTranslations: [String: String]
    let quickFilterTags: [String]?
    let models: [ModelInfo]
    let metadata: Metadata?
    
    struct Metadata: Codable {
        let version: String
        let lastUpdated: String
        let schemaVersion: String
        let totalModels: Int
        let supportedPlatforms: [String]
        let minAppVersion: String
    }
}

//
//  String+Extension.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/14.
//

import Foundation
import CryptoKit

extension String {
    
    /// Generate a stable hash value for temporary file naming
    ///
    /// Unlike the hash property in Swift's standard library, this method
    /// generates the same hash value for identical strings across different
    /// app launches, ensuring that resumable download functionality works correctly.
    ///
    /// - Returns: A stable hash value based on SHA256
    var stableHash: String {
        let data = self.data(using: .utf8) ?? Data()
        let digest = SHA256.hash(data: data)
        return digest.compactMap { String(format: "%02x", $0) }.joined().prefix(16).description
    }
    
    func removingTaobaoPrefix() -> String {
        return self.replacingOccurrences(of: "taobao-mnn/", with: "")
    }
}

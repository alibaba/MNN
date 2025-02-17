//
//  URL+Extension.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/10.
//

import Foundation
import UniformTypeIdentifiers

extension URL {
    func isHEICImage() -> Bool {
        let fileExtension = self.pathExtension.lowercased()
        let utType = UTType(filenameExtension: fileExtension)
        
        return utType == .heif || utType == .heic
    }
}

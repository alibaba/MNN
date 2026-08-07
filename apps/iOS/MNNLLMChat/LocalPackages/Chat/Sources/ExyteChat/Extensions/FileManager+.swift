//
//  FileManager+.swift
//  
//
//  Created by Alisa Mylnikova on 10.03.2023.
//

import Foundation

extension FileManager {

    static var tempDirPath: URL {
        URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
    }

    static var tempFile: URL {
        FileManager.tempDirPath.appendingPathComponent(UUID().uuidString)
    }

    static var tempAudioFile: URL {
        FileManager.tempDirPath.appendingPathComponent(UUID().uuidString + ".aac")
    }
}

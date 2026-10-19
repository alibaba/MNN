//
//  Attachment+Extension.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/17.
//

import ExyteChat

public extension Attachment {
    var mediaString: String {
        switch type {
        case .image:
            return "\(full.path)"
        case .video:
            return "\(full.path)"
        }
    }
}

public extension Recording {
    var recordingString: String {
        return "\(String(describing: url?.path))"
    }

    var durationString: String {
        return "\(String(duration))"
    }
}

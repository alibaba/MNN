//
//  Attachment+Extension.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/17.
//

import ExyteChat

extension Attachment {
    public var mediaString: String {
        switch type {
        case .image:
            return "\(full.path)"
        case .video:
            return "\(full.path)"
        }
    }
}

extension Recording {
    public var recordingString: String {
        return "\(String(describing: url?.path))"
    }
    
    public var durationString: String {
        return "\(String(duration))"
    }
}

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
        // 可以根据需要添加其他类型的处理
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

//
//  SelectionParamsHolder.swift
//  
//
//  Created by Alisa Mylnikova on 05.05.2023.
//

import SwiftUI

final public class SelectionParamsHolder: ObservableObject {

    @Published public var mediaType: MediaSelectionType = .photoAndVideo
    @Published public var selectionStyle: MediaSelectionStyle = .checkmark
    @Published public var selectionLimit: Int? // if nil - unlimited
    @Published public var showFullscreenPreview: Bool = true // if false, tap on image immediately selects this image and closes the picker

    public init(mediaType: MediaSelectionType = .photoAndVideo, selectionStyle: MediaSelectionStyle = .checkmark, selectionLimit: Int? = nil, showFullscreenPreview: Bool = true) {
        self.mediaType = mediaType
        self.selectionStyle = selectionStyle
        self.selectionLimit = selectionLimit
        self.showFullscreenPreview = showFullscreenPreview
    }
}

public enum MediaSelectionStyle {
    case checkmark
    case count
}

public enum MediaSelectionType {
    case photoAndVideo
    case photo
    case video

    var allowsPhoto: Bool {
        [.photoAndVideo, .photo].contains(self)
    }

    var allowsVideo: Bool {
        [.photoAndVideo, .video].contains(self)
    }
}

//
//  CameraSelectionService.swift
//  
//
//  Created by Alisa Mylnikova on 12.07.2022.
//

import SwiftUI

final class CameraSelectionService: ObservableObject {

    var mediaSelectionLimit: Int? // if nill - unlimited
    var onChange: MediaPickerCompletionClosure? = nil

    @Published private(set) var added: [URLMediaModel] = []
    @Published private(set) var selected: [URLMediaModel] = []

    var hasSelected: Bool {
        !selected.isEmpty
    }

    var fitsSelectionLimit: Bool {
        if let selectionLimit = mediaSelectionLimit {
            return selected.count < selectionLimit
        }
        return true
    }

    func canSelect(media: URLMediaModel) -> Bool {
        fitsSelectionLimit || selected.contains(media)
    }

    func onSelect(media: URLMediaModel) {
        if added.contains(media) {
            if let index = selected.firstIndex(of: media) {
                selected.remove(at: index)
            } else if fitsSelectionLimit {
                selected.append(media)
            }
        } else {
            added.append(media)
            if fitsSelectionLimit {
                selected.append(media)
            }
        }
        onChange?(mapToMedia())
    }

    func onSelect(index: Int) {
        guard added.indices.contains(index) else { return }
        let media = added[index]
        if let index = selected.firstIndex(of: media) {
            selected.remove(at: index)
        } else if fitsSelectionLimit {
            selected.append(media)
        }
        onChange?(mapToMedia())
    }

    func isSelected(index: Int) -> Bool {
        guard added.indices.contains(index) else { return false }
        return selected.contains(added[index])
    }

    func selectedIndex(fromAddedIndex index: Int) -> Int? {
        guard added.indices.contains(index) else { return nil }
        let media = added[index]
        return selected.firstIndex(of: media)
    }

    func mapToMedia() -> [Media] {
        selected
            .compactMap {
                guard $0.mediaType != nil else {
                    return nil
                }
                return Media(source: $0)
            }
    }

    func removeAll() {
        selected.removeAll()
        added.removeAll()
        onChange?([])
    }
}

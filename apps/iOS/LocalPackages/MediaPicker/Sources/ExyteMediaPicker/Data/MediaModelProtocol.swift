//
//  SwiftUIView.swift
//  
//
//  Created by Alisa Mylnikova on 21.04.2023.
//

import SwiftUI

protocol MediaModelProtocol {
    var mediaType: MediaType? { get }
    var duration: CGFloat? { get }

    func getURL() async -> URL?
    func getThumbnailURL() async -> URL?

    func getData() async throws -> Data?
    func getThumbnailData() async -> Data?
}

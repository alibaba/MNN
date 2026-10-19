//
//  Created by Alex.M on 06.06.2022.
//

import Foundation
import SwiftUI

public struct MediasGrid<Data, Camera, Content, LoadingCell>: View where Data: RandomAccessCollection, Data.Element: Identifiable, Camera: View, Content: View, LoadingCell: View {

    public let data: Data
    public let camera: () -> Camera
    public let content: (Data.Element) -> Content
    public let loadingCell: () -> LoadingCell
    
    @Environment(\.mediaPickerTheme) private var theme

    private var columns: [GridItem] {
        [GridItem(.adaptive(minimum: 100), spacing: theme.cellStyle.columnsSpacing, alignment: .top)]
    }
    
    public init(_ data: Data, @ViewBuilder camera: @escaping () -> Camera, @ViewBuilder content: @escaping (Data.Element) -> Content, @ViewBuilder loadingCell: @escaping () -> LoadingCell) {
        self.data = data
        self.camera = camera
        self.content = content
        self.loadingCell = loadingCell
    }

    public var body: some View {
        LazyVGrid(columns: columns, spacing: theme.cellStyle.rowSpacing) {
            camera()
            ForEach(data) { item in
                content(item)
            }
            loadingCell()
        }
    }
}

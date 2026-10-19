//
//  Created by Alex.M on 06.07.2022.
//

import Foundation
import SwiftUI

struct MediaPickerThemeKey: EnvironmentKey {
    static var defaultValue: MediaPickerTheme = MediaPickerTheme()
}

extension EnvironmentValues {
    public var mediaPickerTheme: MediaPickerTheme {
        get { self[MediaPickerThemeKey.self] }
        set { self[MediaPickerThemeKey.self] = newValue }
    }
}

public extension View {
    func mediaPickerTheme(_ theme: MediaPickerTheme) -> some View {
        self.environment(\.mediaPickerTheme, theme)
    }

    func mediaPickerTheme(
        main: MediaPickerTheme.Main = .init(),
        selection: MediaPickerTheme.Selection = .init(),
        cellStyle: MediaPickerTheme.CellStyle = .init(),
        error: MediaPickerTheme.Error = .init(),
        defaultHeader: MediaPickerTheme.DefaultHeader = .init()
    ) -> some View {
        self.environment(\.mediaPickerTheme, MediaPickerTheme(main: main, selection: selection, cellStyle: cellStyle, error: error, defaultHeader: defaultHeader))
    }
}

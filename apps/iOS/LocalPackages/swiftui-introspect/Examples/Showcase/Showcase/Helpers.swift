import SwiftUI

extension View {
    /// Modify a view with a `ViewBuilder` closure.
    ///
    /// This represents a streamlining of the
    /// [`modifier`](https://developer.apple.com/documentation/swiftui/view/modifier(_:)) +
    /// [`ViewModifier`](https://developer.apple.com/documentation/swiftui/viewmodifier) pattern.
    ///
    /// - Note: Useful only when you don't need to reuse the closure.
    /// If you do, turn the closure into a proper modifier.
    public func modifier<ModifiedContent: View>(
        @ViewBuilder _ modifier: (Self) -> ModifiedContent
    ) -> ModifiedContent {
        modifier(self)
    }
}

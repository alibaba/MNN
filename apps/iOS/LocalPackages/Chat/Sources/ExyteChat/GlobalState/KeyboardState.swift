//
//  Created by Alex.M on 02.10.2023.
//

import Foundation
import Combine
import UIKit

public final class KeyboardState: ObservableObject {
    @Published private(set) public var isShown: Bool = false

    private var subscriptions = Set<AnyCancellable>()

    init() {
        subscribeKeyboardNotifications()
    }
}

private extension KeyboardState {
    func subscribeKeyboardNotifications() {
        Publishers.Merge(
            NotificationCenter.default
                .publisher(for: UIResponder.keyboardWillShowNotification)
                .map { _ in true },

            NotificationCenter.default
                .publisher(for: UIResponder.keyboardWillHideNotification)
                .map { _ in false }
        )
        .receive(on: RunLoop.main)
        .assign(to: \.isShown, on: self)
        .store(in: &subscriptions)
    }
}

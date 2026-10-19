//
//  Created by Alex.M on 07.06.2022.
//

import Foundation
import Combine

extension Set where Element == AnyCancellable {
    mutating func cancelAll() {
        self = Set<AnyCancellable>()
    }
}

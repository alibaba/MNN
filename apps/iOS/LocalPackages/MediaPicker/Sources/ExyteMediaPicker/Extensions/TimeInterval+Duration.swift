//
//  Created by Alex.M on 31.05.2022.
//

import Foundation

extension TimeInterval {
    func formatted(locale: Locale = .current) -> String? {
        let formatter = DateComponentsFormatter()
        formatter.unitsStyle = .abbreviated
        formatter.zeroFormattingBehavior = .dropAll
        formatter.allowedUnits = [.day, .hour, .minute, .second]
        formatter.maximumUnitCount = 2

        formatter.calendar = Calendar.current
        formatter.calendar?.locale = locale

        return formatter.string(from: self)
    }
}

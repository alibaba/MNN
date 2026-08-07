//
//  Created by Alex.M on 08.07.2022.
//

import Foundation

extension Date {
    func randomTime() -> Date {
        var hour = Int.random(min: 0, max: 23)
        var minute = Int.random(min: 0, max: 59)
        var second = Int.random(min: 0, max: 59)

        let current = Calendar.current.dateComponents([.hour, .minute, .second], from: Date())
        let curHour = current.hour ?? 23
        let curMinute = current.minute ?? 59
        let curSecond = current.second ?? 59

        if hour > curHour {
            hour = curHour
        } else if hour == curHour, minute > curMinute {
            minute = curMinute
        } else if hour == curHour, minute == curMinute, second > curSecond {
            second = curSecond
        }

        var components = Calendar.current.dateComponents([.year, .month, .day], from: self)
        components.hour = hour
        components.minute = minute
        components.second = second
        return Calendar.current.date(from: components)!
    }
}

class DateFormatting {
    static var agoFormatter = RelativeDateTimeFormatter()
}

extension Date {
    // 1 hour ago, 2 days ago...
    func formatAgo() -> String {
        let result = DateFormatting.agoFormatter.localizedString(for: self, relativeTo: Date())
        if result.contains("second") {
            return "Just now"
        }
        return result
    }
}

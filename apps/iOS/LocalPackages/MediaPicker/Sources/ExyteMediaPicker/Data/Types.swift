//
//  Created by Alex.M on 04.07.2022.
//

import Foundation

public typealias MediaPickerCompletionClosure = ([Media]) -> Void
public typealias MediaPickerOrientationHandler = (ShouldLock) -> Void
public typealias SimpleClosure = ()->()

public enum ShouldLock {
    case lock, unlock
}

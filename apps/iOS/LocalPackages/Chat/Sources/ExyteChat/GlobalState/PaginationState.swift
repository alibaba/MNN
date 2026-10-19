//
//  Created by Alex.M on 30.06.2022.
//

import Foundation

public typealias ChatPaginationClosure = (Message) async -> Void

final class PaginationHandler: ObservableObject {
    var handleClosure: ChatPaginationClosure
    var pageSize: Int

    init(handleClosure: @escaping ChatPaginationClosure, pageSize: Int) {
        self.handleClosure = handleClosure
        self.pageSize = pageSize
    }

    func handle(_ message: Message) async {
        await handleClosure(message)
    }
}

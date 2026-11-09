//
//  Created by Alex.M on 09.06.2022.
//

import Foundation
import Combine

protocol AlbumsProviderProtocol {
    var albums: AnyPublisher<[AlbumModel], Never> { get }

    func reload()
}

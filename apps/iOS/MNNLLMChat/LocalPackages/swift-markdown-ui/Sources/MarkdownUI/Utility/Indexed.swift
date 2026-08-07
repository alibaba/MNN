import Foundation

struct Indexed<Value> {
  let index: Int
  let value: Value
}

extension Indexed: Equatable where Value: Equatable {}
extension Indexed: Hashable where Value: Hashable {}

extension Sequence {
  func indexed() -> [Indexed<Element>] {
    zip(0..., self).map { index, value in
      Indexed(index: index, value: value)
    }
  }
}

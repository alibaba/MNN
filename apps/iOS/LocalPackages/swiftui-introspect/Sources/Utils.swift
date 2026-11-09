postfix operator ~

postfix func ~ <LHS, T>(lhs: LHS) -> T {
    lhs as! T
}

postfix func ~ <LHS, T>(lhs: LHS?) -> T? {
    lhs as? T
}

func recursiveSequence<S: Sequence>(_ sequence: S, children: @escaping (S.Element) -> S) -> AnySequence<S.Element> {
    AnySequence {
        var mainIterator = sequence.makeIterator()
        // Current iterator, or `nil` if all sequences are exhausted:
        var iterator: AnyIterator<S.Element>?

        return AnyIterator {
            guard let iterator, let element = iterator.next() else {
                if let element = mainIterator.next() {
                    iterator = recursiveSequence(children(element), children: children).makeIterator()
                    return element
                }
                return nil
            }
            return element
        }
    }
}

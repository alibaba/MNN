//
//  Created by ktiays on 2025/1/31.
//  Copyright (c) 2025 ktiays. All rights reserved.
//

import DequeModule
import UIKit

private class ObjectPool<T: Equatable & Hashable> {
    private let factory: () -> T
    fileprivate lazy var objects: Deque<T> = .init()

    public init(_ factory: @escaping () -> T) {
        self.factory = factory
    }

    open func acquire() -> T {
        if let object = objects.popFirst() {
            return object
        } else {
            let object = factory()
            return object
        }
    }

    open func stash(_ object: T) {
        objects.append(object)
    }

    open func reorder(matching sequence: [T]) {
        var current = Set(objects)
        objects.removeAll()
        for content in sequence where current.contains(content) {
            objects.append(content)
            current.remove(content)
        }
        for reset in current {
            objects.append(reset) // stash the rest
        }
    }
}

private class ViewBox<T: UIView>: ObjectPool<T> {
    override func acquire() -> T {
        super.acquire()
    }

    override func stash(_ item: T) {
        super.stash(item)
    }
}

public final class ReusableViewProvider {
    private let codeViewPool: ViewBox<CodeView> = .init {
        CodeView()
    }

    private let tableViewPool: ViewBox<TableView> = .init {
        TableView()
    }

    private let lock = NSLock()

    public init() {}

    func lockPool() {
        lock.lock()
    }

    func unlockPool() {
        lock.unlock()
    }

    func removeAll() {
        codeViewPool.objects.removeAll()
        tableViewPool.objects.removeAll()
    }

    func acquireCodeView() -> CodeView {
        codeViewPool.acquire()
    }

    func stashCodeView(_ codeView: CodeView) {
        codeViewPool.stash(codeView)
    }

    func acquireTableView() -> TableView {
        tableViewPool.acquire()
    }

    func stashTableView(_ tableView: TableView) {
        tableViewPool.stash(tableView)
    }

    func reorderViews(matching sequence: [UIView]) {
        // we adjust the sequence of stashed views to match the order
        // afterwards when TextBuilder visit a node requesting new view
        // it will follow the order to avoid glitch

        let orderedCodeView = sequence.compactMap { $0 as? CodeView }
        let orderedTableView = sequence.compactMap { $0 as? TableView }

        codeViewPool.reorder(matching: orderedCodeView)
        tableViewPool.reorder(matching: orderedTableView)
    }
}

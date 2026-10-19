//
//  LTXLabel+UIContextMenuInteractionDelegate.swift
//  MarkdownView
//
//  Created by 秋星桥 on 7/8/25.
//

import UIKit

extension LTXLabel: UIContextMenuInteractionDelegate {
    public func contextMenuInteraction(
        _: UIContextMenuInteraction,
        configurationForMenuAtLocation location: CGPoint
    ) -> UIContextMenuConfiguration? {
        #if targetEnvironment(macCatalyst)
            guard selectionRange != nil else { return nil }
            var menuItems: [UIMenuElement] = [
                UIAction(title: LocalizedText.copy, image: nil) { _ in
                    self.copySelectedText()
                },
            ]
            if selectionRange != selectAllRange() {
                menuItems.append(
                    UIAction(title: LocalizedText.selectAll, image: nil) { _ in
                        self.selectAllText()
                    }
                )
            }
            return .init(
                identifier: nil,
                previewProvider: nil
            ) { _ in
                .init(children: menuItems)
            }
        #else
            DispatchQueue.main.async {
                guard self.isSelectable else { return }
                guard self.isLocationInSelection(location: location) else { return }
                self.showSelectionMenuController()
            }
            return nil
        #endif
    }
}

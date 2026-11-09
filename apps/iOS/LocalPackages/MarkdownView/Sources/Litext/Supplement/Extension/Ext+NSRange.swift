//
//  Ext+NSRange.swift
//  Litext
//
//  Created by 秋星桥 on 3/26/25.
//

import Foundation

extension NSRange {
    func contains(_ index: Int) -> Bool {
        index >= location && index < (location + length)
    }
}

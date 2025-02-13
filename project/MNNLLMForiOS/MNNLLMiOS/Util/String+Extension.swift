//
//  String+Extension.swift
//  MNNLLMiOS
//
//  Created by 游薪渝(揽清) on 2025/1/14.
//

import Foundation

extension String {
    func removingTaobaoPrefix() -> String {
        return self.replacingOccurrences(of: "taobao-mnn/", with: "")
    }
}

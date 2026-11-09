
//
//  Created by Mike Griebling on 2022-12-31.
//  Translated from an Objective-C implementation by Markus Sähn.
//
//  This software may be modified and distributed under the terms of the
//  MIT license. See the LICENSE file for details.
//

import Foundation

extension MTColor {
    
    public convenience init?(fromHexString hexString:String) {
        if hexString.isEmpty { return nil }
        if !hexString.hasPrefix("#") { return nil }
        
        var rgbValue = UInt64(0)
        let scanner = Scanner(string: hexString)
        scanner.charactersToBeSkipped = CharacterSet(charactersIn: "#")
        scanner.scanHexInt64(&rgbValue)
        self.init(red: CGFloat((rgbValue & 0xFF0000) >> 16)/255.0,
                  green: CGFloat((rgbValue & 0xFF00) >> 8)/255.0,
                  blue: CGFloat((rgbValue & 0xFF))/255.0,
                  alpha: 1.0)
    }
    
}

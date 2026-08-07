import Foundation

//
//  Created by Mike Griebling on 2022-12-31.
//  Translated from an Objective-C implementation by 安志钢.
//
//  This software may be modified and distributed under the terms of the
//  MIT license. See the LICENSE file for details.
//

#if os(iOS) || os(visionOS)

import UIKit

public typealias MTView = UIView
public typealias MTColor = UIColor
public typealias MTBezierPath = UIBezierPath
public typealias MTLabel = UILabel
public typealias MTEdgeInsets = UIEdgeInsets
public typealias MTRect = CGRect
public typealias MTImage = UIImage

let MTEdgeInsetsZero = UIEdgeInsets.zero
func MTGraphicsGetCurrentContext() -> CGContext? { UIGraphicsGetCurrentContext() }

#else

import AppKit

public typealias MTView = NSView
public typealias MTColor = NSColor
public typealias MTBezierPath = NSBezierPath
public typealias MTEdgeInsets = NSEdgeInsets
public typealias MTRect = NSRect
public typealias MTImage = NSImage

let MTEdgeInsetsZero = NSEdgeInsets.init(top: 0, left: 0, bottom: 0, right: 0)
func MTGraphicsGetCurrentContext() -> CGContext? { NSGraphicsContext.current?.cgContext }

#endif

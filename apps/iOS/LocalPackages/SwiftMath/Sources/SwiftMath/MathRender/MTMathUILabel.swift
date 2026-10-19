//
//  Created by Mike Griebling on 2022-12-31.
//  Translated from an Objective-C implementation by Kostub Deshmukh.
//
//  This software may be modified and distributed under the terms of the
//  MIT license. See the LICENSE file for details.
//

import Foundation
import CoreText

/**
 Different display styles supported by the `MTMathUILabel`.
 
 The only significant difference between the two modes is how fractions
 and limits on large operators are displayed.
 */
public enum MTMathUILabelMode {
    /// Display mode. Equivalent to $$ in TeX
    case display
    /// Text mode. Equivalent to $ in TeX.
    case text
}

/**
    Horizontal text alignment for `MTMathUILabel`.
 */
public enum MTTextAlignment : UInt {
    /// Align left.
    case left
    /// Align center.
    case center
    /// Align right.
    case right
}

/** The main view for rendering math.
 
 `MTMathLabel` accepts either a string in LaTeX or an `MTMathList` to display. Use
 `MTMathList` directly only if you are building it programmatically (e.g. using an
 editor), otherwise using LaTeX is the preferable method.
 
 The math display is centered vertically in the label. The default horizontal alignment is
 is left. This can be changed by setting `textAlignment`. The math is default displayed in
 *Display* mode. This can be changed using `labelMode`.
 
 When created it uses `[MTFontManager defaultFont]` as its font. This can be changed using
 the `font` parameter.
 */
@IBDesignable
public class MTMathUILabel : MTView {
        
    /** The `MTMathList` to render. Setting this will remove any
     `latex` that has already been set. If `latex` has been set, this will
     return the parsed `MTMathList` if the `latex` parses successfully. Use this
     setting if the `MTMathList` has been programmatically constructed, otherwise it
     is preferred to use `latex`.
     */
    public var mathList:MTMathList? {
        set {
            _mathList = newValue
            _error = nil
            _latex = MTMathListBuilder.mathListToString(newValue)
            self.invalidateIntrinsicContentSize()
            self.setNeedsLayout()
        }
        get { _mathList }
    }
    private var _mathList:MTMathList?
    
    /** The latex string to be displayed. Setting this will remove any `mathList` that
     has been set. If latex has not been set, this will return the latex output for the
     `mathList` that is set.
     @see error */
    @IBInspectable
    public var latex:String {
        set {
            _latex = newValue
            _error = nil
            var error : NSError? = nil
            _mathList = MTMathListBuilder.build(fromString: newValue, error: &error)
            if error != nil {
                _mathList = nil
                _error = error
                self.errorLabel?.text = error!.localizedDescription
                self.errorLabel?.frame = self.bounds
                self.errorLabel?.isHidden = !self.displayErrorInline
            } else {
                self.errorLabel?.isHidden = true
            }
            self.invalidateIntrinsicContentSize()
            self.setNeedsLayout()
        }
        get { _latex }
    }
    private var _latex = ""
    
    /** This contains any error that occurred when parsing the latex. */
    public var error:NSError? { _error }
    private var _error:NSError?
    
    /** If true, if there is an error it displays the error message inline. Default true. */
    public var displayErrorInline = true
    
    /** The MTFont to use for rendering. */
    public var font:MTFont? {
        set {
            guard newValue != nil else { return }
            _font = newValue
            self.invalidateIntrinsicContentSize()
            self.setNeedsLayout()
        }
        get { _font }
    }
    private var _font:MTFont?
    
    /** Convenience method to just set the size of the font without changing the fontface. */
    @IBInspectable
    public var fontSize:CGFloat {
        set {
            _fontSize = newValue
            let font = font?.copy(withSize: newValue)
            self.font = font  // also forces an update
        }
        get { _fontSize }
    }
    private var _fontSize:CGFloat=0
    
    /** This sets the text color of the rendered math formula. The default color is black. */
    @IBInspectable
    public var textColor:MTColor? {
        set {
            guard newValue != nil else { return }
            _textColor = newValue
            self.displayList?.textColor = newValue
            self.setNeedsDisplay()
        }
        get { _textColor }
    }
    private var _textColor:MTColor?
    
    /** The minimum distance from the margin of the view to the rendered math. This value is
     `UIEdgeInsetsZero` by default. This is useful if you need some padding between the math and
     the border/background color. sizeThatFits: will have its returned size increased by these insets.
     */
    @IBInspectable
    public var contentInsets:MTEdgeInsets {
        set {
            _contentInsets = newValue
            self.invalidateIntrinsicContentSize()
            self.setNeedsLayout()
        }
        get { _contentInsets }
    }
    private var _contentInsets = MTEdgeInsetsZero
    
    /** The Label mode for the label. The default mode is Display */
    public var labelMode:MTMathUILabelMode {
        set {
            _labelMode = newValue
            self.invalidateIntrinsicContentSize()
            self.setNeedsLayout()
        }
        get { _labelMode }
    }
    private var _labelMode = MTMathUILabelMode.display
    
    /** Horizontal alignment for the text. The default is align left. */
    public var textAlignment:MTTextAlignment {
        set {
            _textAlignment = newValue
            self.invalidateIntrinsicContentSize()
            self.setNeedsLayout()
        }
        get { _textAlignment }
    }
    private var _textAlignment = MTTextAlignment.left
    
    /** The internal display of the MTMathUILabel. This is for advanced use only. */
    public var displayList: MTMathListDisplay? { _displayList }
    private var _displayList:MTMathListDisplay?
    
    public var currentStyle:MTLineStyle {
        switch _labelMode {
            case .display: return .display
            case .text: return .text
        }
    }
    
    public var errorLabel: MTLabel?
    
    public override init(frame: CGRect) {
        super.init(frame: frame)
        self.initCommon()
    }
    
    public required init?(coder: NSCoder) {
        super.init(coder: coder)
        self.initCommon()
    }
    
    func initCommon() {
#if os(macOS)
        self.layer?.isGeometryFlipped = true
#else
        self.layer.isGeometryFlipped = true
#endif
        _fontSize = 20
        _contentInsets = MTEdgeInsetsZero
        _labelMode = .display
        let font = MTFontManager.fontManager.defaultFont
        self.font = font
        _textAlignment = .left
        _displayList = nil
        displayErrorInline = true
        self.backgroundColor = MTColor.clear
        
        _textColor = MTColor.black
        let label = MTLabel()
        self.errorLabel = label
#if os(macOS)
        label.layer?.isGeometryFlipped = true
#else
        label.layer.isGeometryFlipped = true
#endif
        label.isHidden = true
        label.textColor = MTColor.red
        self.addSubview(label)
    }
    
    override public func draw(_ dirtyRect: MTRect) {
        super.draw(dirtyRect)
        if self.mathList == nil { return }

        // drawing code
        let context = MTGraphicsGetCurrentContext()!
        context.saveGState()
        displayList!.draw(context)
        context.restoreGState()
    }
    
    func _layoutSubviews() {
        if _mathList != nil {
            // print("Pre list = \(_mathList!)")
            _displayList = MTTypesetter.createLineForMathList(_mathList, font: font, style: currentStyle)
            _displayList!.textColor = textColor
            // print("Post list = \(_mathList!)")
            var textX = CGFloat(0)
            switch self.textAlignment {
                case .left:   textX = contentInsets.left
                case .center: textX = (bounds.size.width - contentInsets.left - contentInsets.right - _displayList!.width) / 2 + contentInsets.left
                case .right:  textX = bounds.size.width - _displayList!.width - contentInsets.right
            }
            let availableHeight = bounds.size.height - contentInsets.bottom - contentInsets.top
            
            // center things vertically
            var height = _displayList!.ascent + _displayList!.descent
            if height < fontSize/2 {
                height = fontSize/2  // set height to half the font size
            }
            let textY = (availableHeight - height) / 2 + _displayList!.descent + contentInsets.bottom
            _displayList!.position = CGPointMake(textX, textY)
        } else {
            _displayList = nil
        }
        errorLabel?.frame = self.bounds
        self.setNeedsDisplay()
    }
    
    func _sizeThatFits(_ size:CGSize) -> CGSize {
        guard _mathList != nil else { return size }
        var size = size
        var displayList:MTMathListDisplay? = nil
        displayList = MTTypesetter.createLineForMathList(_mathList, font: font, style: currentStyle)
        size.width = displayList!.width + contentInsets.left + contentInsets.right
        size.height = displayList!.ascent + displayList!.descent + contentInsets.top + contentInsets.bottom
        return size
    }
    
#if os(macOS)
    func setNeedsDisplay() { self.needsDisplay = true }
    func setNeedsLayout() { self.needsLayout = true }
    public override var fittingSize: CGSize { _sizeThatFits(CGSizeZero) }
    override public var isFlipped: Bool { false }
    override public func layout() {
        self._layoutSubviews()
        super.layout()
    }
#else
    public override var intrinsicContentSize: CGSize { _sizeThatFits(CGSizeZero) }
    override public func layoutSubviews() { _layoutSubviews() }
#endif
    
}

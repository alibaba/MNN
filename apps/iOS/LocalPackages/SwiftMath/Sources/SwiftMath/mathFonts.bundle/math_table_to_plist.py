#!/usr/local/bin/python3
import plistlib
import sys
from fontTools.ttLib import TTFont

def usage(code):
    print('Usage math_table_to_plist.py <fontfile> <plistfile>')
    sys.exit(code)

def process_font(font_file, out_file):
    font = TTFont(font_file)
    math_table = font['MATH'].table
    constants = get_constants(math_table)
    italic_c = get_italic_correction(math_table)
    v_variants = get_v_variants(math_table)
    h_variants = get_h_variants(math_table)
    assembly = get_v_assembly(math_table)
    accents = get_accent_attachments(math_table)
    pl = {
            "version" : "1.3",
            "constants": constants,
            "v_variants" : v_variants,
            "h_variants" : h_variants,
            "italic" : italic_c,
            "accents" : accents,
            "v_assembly" : assembly }
    ofile = open(out_file, 'w+b')
    plistlib.dump(pl, ofile)
    ofile.close()

def get_constants(math_table):
    constants = math_table.MathConstants
    if constants is None:
        raise 'Cannot find MathConstants in MATH table'

    int_consts = [ 'ScriptPercentScaleDown',
            'ScriptScriptPercentScaleDown',
            'DelimitedSubFormulaMinHeight',
            'DisplayOperatorMinHeight',
            'RadicalDegreeBottomRaisePercent']
    consts = { c : getattr(constants, c) for c in int_consts }

    record_consts = [ 'MathLeading',
            'AxisHeight',
            'AccentBaseHeight',
            'FlattenedAccentBaseHeight',
            'SubscriptShiftDown',
            'SubscriptTopMax',
            'SubscriptBaselineDropMin',
            'SuperscriptShiftUp',
            'SuperscriptShiftUpCramped',
            'SuperscriptBottomMin',
            'SuperscriptBaselineDropMax',
            'SubSuperscriptGapMin',
            'SuperscriptBottomMaxWithSubscript',
            'SpaceAfterScript',
            'UpperLimitGapMin',
            'UpperLimitBaselineRiseMin',
            'LowerLimitGapMin',
            'LowerLimitBaselineDropMin',
            'StackTopShiftUp',
            'StackTopDisplayStyleShiftUp',
            'StackBottomShiftDown',
            'StackBottomDisplayStyleShiftDown',
            'StackGapMin',
            'StackDisplayStyleGapMin',
            'StretchStackTopShiftUp',
            'StretchStackBottomShiftDown',
            'StretchStackGapAboveMin',
            'StretchStackGapBelowMin',
            'FractionNumeratorShiftUp',
            'FractionNumeratorDisplayStyleShiftUp',
            'FractionDenominatorShiftDown',
            'FractionDenominatorDisplayStyleShiftDown',
            'FractionNumeratorGapMin',
            'FractionNumDisplayStyleGapMin',
            'FractionRuleThickness',
            'FractionDenominatorGapMin',
            'FractionDenomDisplayStyleGapMin',
            'SkewedFractionHorizontalGap',
            'SkewedFractionVerticalGap',
            'OverbarVerticalGap',
            'OverbarRuleThickness',
            'OverbarExtraAscender',
            'UnderbarVerticalGap',
            'UnderbarRuleThickness',
            'UnderbarExtraDescender',
            'RadicalVerticalGap',
            'RadicalDisplayStyleVerticalGap',
            'RadicalRuleThickness',
            'RadicalExtraAscender',
            'RadicalKernBeforeDegree',
            'RadicalKernAfterDegree',
    ]
    consts_2 = { c : getattr(constants, c).Value for c in record_consts }
    consts.update(consts_2)
    
    variants = math_table.MathVariants
    consts['MinConnectorOverlap'] = variants.MinConnectorOverlap
    return consts

def get_italic_correction(math_table):
    glyph_info = math_table.MathGlyphInfo
    if glyph_info is None:
        raise "Cannot find MathGlyphInfo in MATH table."
    italic = glyph_info.MathItalicsCorrectionInfo
    if italic is None:
        raise "Cannot find Italic Correction in GlyphInfo"

    glyphs = italic.Coverage.glyphs
    count = italic.ItalicsCorrectionCount
    records = italic.ItalicsCorrection
    italic_dict = {}
    for i in range(count):
        name = glyphs[i]
        record = records[i]
        if record.DeviceTable is not None:
            raise "Don't know how to process device table for italic correction."
        italic_dict[name] = record.Value
    return italic_dict

def get_accent_attachments(math_table):
    glyph_info = math_table.MathGlyphInfo
    if glyph_info is None:
        raise "Cannot find MathGlyphInfo in MATH table."
    attach = glyph_info.MathTopAccentAttachment
    if attach is None:
        raise "Cannot find Top Accent Attachment in GlyphInfo"

    glyphs = attach.TopAccentCoverage.glyphs
    count = attach.TopAccentAttachmentCount
    records = attach.TopAccentAttachment
    attach_dict = {}
    for i in range(count):
        name = glyphs[i]
        record = records[i]
        if record.DeviceTable is not None:
            raise "Don't know how to process device table for accent attachment."
        attach_dict[name] = record.Value
    return attach_dict

def get_v_variants(math_table):
    variants = math_table.MathVariants
    vglyphs = variants.VertGlyphCoverage.glyphs
    vconstruction = variants.VertGlyphConstruction
    count = variants.VertGlyphCount
    variant_dict = {}
    for i in range(count):
        name = vglyphs[i]
        record = vconstruction[i]
        glyph_variants = [x.VariantGlyph for x in
                record.MathGlyphVariantRecord]
        variant_dict[name] = glyph_variants
    return variant_dict

def get_h_variants(math_table):
    variants = math_table.MathVariants
    hglyphs = variants.HorizGlyphCoverage.glyphs
    hconstruction = variants.HorizGlyphConstruction
    count = variants.HorizGlyphCount
    variant_dict = {}
    for i in range(count):
        name = hglyphs[i]
        record = hconstruction[i]
        glyph_variants = [x.VariantGlyph for x in
                record.MathGlyphVariantRecord]
        variant_dict[name] = glyph_variants
    return variant_dict

def get_v_assembly(math_table):
    variants = math_table.MathVariants
    vglyphs = variants.VertGlyphCoverage.glyphs
    vconstruction = variants.VertGlyphConstruction
    count = variants.VertGlyphCount
    assembly_dict = {}
    for i in range(count):
        name = vglyphs[i]
        record = vconstruction[i]
        assembly = record.GlyphAssembly
        if assembly is not None:
            # There is an assembly for this glyph
            italic = assembly.ItalicsCorrection.Value
            parts = [part_dict(part) for part in assembly.PartRecords]
            assembly_dict[name] = { 
                    "italic" : assembly.ItalicsCorrection.Value,
                    "parts" : parts }
    return assembly_dict

def part_dict(part):
    return {
            "glyph": part.glyph,
            "startConnector" : part.StartConnectorLength,
            "endConnector" : part.EndConnectorLength,
            "advance" : part.FullAdvance,
            "extender" : (part.PartFlags == 1) }

def main():
    if len(sys.argv) != 3:
        usage(1)        
    font_file = sys.argv[1]
    plist_file = sys.argv[2]
    process_font(font_file, plist_file)

if __name__ == '__main__':
    main()

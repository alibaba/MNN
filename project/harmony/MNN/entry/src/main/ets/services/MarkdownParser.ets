export enum MarkdownBlockType {
  PARAGRAPH = 'paragraph',
  HEADING = 'heading',
  CODE = 'code',
  QUOTE = 'quote',
  LIST = 'list',
  TABLE = 'table',
  MATH = 'math'
}

export class MarkdownBlock {
  id: number = 0;
  type: MarkdownBlockType = MarkdownBlockType.PARAGRAPH;
  content: string = '';
}

const SUPERSCRIPTS: Record<string, string> = {
  '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
  '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾', 'n': 'ⁿ', 'i': 'ⁱ', 'x': 'ˣ', 'T': 'ᵀ', 'k': 'ᵏ',
  'd': 'ᵈ', 'm': 'ᵐ', 's': 'ˢ', 'r': 'ʳ', 't': 'ᵗ', 'h': 'ʰ', 'a': 'ᵃ', 'b': 'ᵇ', 'c': 'ᶜ', 'e': 'ᵉ',
  'f': 'ᶠ', 'g': 'ᵍ', 'l': 'ˡ', 'o': 'ᵒ', 'p': 'ᵖ', 'u': 'ᵘ', 'v': 'ᵛ', 'w': 'ʷ', 'y': 'ʸ', 'z': 'ᶻ'
};

const SUBSCRIPTS: Record<string, string> = {
  '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
  '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎', 'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ',
  'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ', 'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ',
  'v': 'ᵥ', 'x': 'ₓ'
};

const LATEX_SYMBOLS: [RegExp, string][] = [
  [/\\alpha/g, 'α'], [/\\beta/g, 'β'], [/\\gamma/g, 'γ'], [/\\Gamma/g, 'Γ'], [/\\delta/g, 'δ'],
  [/\\Delta/g, 'Δ'], [/\\epsilon/g, 'ε'], [/\\varepsilon/g, 'ε'], [/\\zeta/g, 'ζ'], [/\\eta/g, 'η'],
  [/\\theta/g, 'θ'], [/\\Theta/g, 'Θ'], [/\\iota/g, 'ι'], [/\\kappa/g, 'κ'], [/\\lambda/g, 'λ'],
  [/\\Lambda/g, 'Λ'], [/\\mu/g, 'μ'], [/\\nu/g, 'ν'], [/\\xi/g, 'ξ'], [/\\Xi/g, 'Ξ'], [/\\pi/g, 'π'],
  [/\\Pi/g, 'Π'], [/\\rho/g, 'ρ'], [/\\sigma/g, 'σ'], [/\\Sigma/g, 'Σ'], [/\\tau/g, 'τ'],
  [/\\upsilon/g, 'υ'], [/\\phi/g, 'φ'], [/\\Phi/g, 'Φ'], [/\\chi/g, 'χ'], [/\\psi/g, 'ψ'],
  [/\\Psi/g, 'Ψ'], [/\\omega/g, 'ω'], [/\\Omega/g, 'Ω'], [/\\hbar/g, 'ℏ'], [/\\ell/g, 'ℓ'],
  [/\\partial/g, '∂'], [/\\nabla/g, '∇'], [/\\infty/g, '∞'], [/\\sum/g, '∑'], [/\\prod/g, '∏'],
  [/\\oint/g, '∮'], [/\\int/g, '∫'], [/\\times/g, '×'], [/\\div/g, '÷'], [/\\cdot/g, '·'],
  [/\\pm/g, '±'], [/\\mp/g, '∓'], [/\\leq/g, '≤'], [/\\geq/g, '≥'], [/\\neq/g, '≠'],
  [/\\approx/g, '≈'], [/\\equiv/g, '≡'], [/\\sim/g, '∼'], [/\\propto/g, '∝'], [/\\rightarrow/g, '→'],
  [/\\leftarrow/g, '←'], [/\\Rightarrow/g, '⇒'], [/\\Leftarrow/g, '⇐'], [/\\leftrightarrow/g, '↔'],
  [/\\to/g, '→'], [/\\in/g, '∈'], [/\\notin/g, '∉'], [/\\subseteq/g, '⊆'], [/\\subset/g, '⊂'],
  [/\\cup/g, '∪'], [/\\cap/g, '∩'], [/\\forall/g, '∀'], [/\\exists/g, '∃'], [/\\emptyset/g, '∅'],
  [/\\angle/g, '∠'], [/\\circ/g, '∘'], [/\\bullet/g, '•'], [/\\ldots/g, '…'], [/\\cdots/g, '⋯'],
  [/\\dots/g, '…'], [/\\langle/g, '⟨'], [/\\rangle/g, '⟩'], [/\\perp/g, '⊥'], [/\\parallel/g, '∥'],
  [/\\otimes/g, '⊗'], [/\\oplus/g, '⊕'], [/\\odot/g, '⊙'], [/\\star/g, '⋆'], [/\\prime/g, '′'],
  [/\\degree/g, '°']
];

export class MarkdownParser {
  /**
   * Renders LaTeX math as Unicode text (Greek letters, operators, super/subscripts, fractions
   * and roots). Mirrors the role of Android's jlatexmath without a full layout engine.
   */
  static latexToUnicode(source: string): string {
    let s = source;
    s = s.replace(/\\text\s*\{([^{}]*)\}/g, '$1');
    s = s.replace(/\\(mathrm|mathbf|mathit|mathcal|operatorname)\s*\{([^{}]*)\}/g, '$2');
    for (let index = 0; index < LATEX_SYMBOLS.length; index++) {
      s = s.replace(LATEX_SYMBOLS[index][0], LATEX_SYMBOLS[index][1]);
    }
    // One nesting level inside the braces is enough for typical model output (x^{n+1} etc.).
    const groupPattern = '((?:[^{}]|\\{[^{}]*\\})*)';
    const fractionPattern = new RegExp(`\\\\frac\\s*\\{${groupPattern}\\}\\s*\\{${groupPattern}\\}`);
    const rootPattern = new RegExp(`\\\\sqrt\\s*\\{${groupPattern}\\}`);
    for (let guard = 0; guard < 64; guard++) {
      const fraction = s.match(fractionPattern);
      if (fraction) {
        s = s.replace(fraction[0], `(${fraction[1]})/(${fraction[2]})`);
        continue;
      }
      const root = s.match(rootPattern);
      if (root) {
        s = s.replace(root[0], `√(${root[1]})`);
        continue;
      }
      break;
    }
    s = s.replace(/\^\{([^{}]*)\}/g, (whole: string, group: string) => MarkdownParser.mapExponents(group, SUPERSCRIPTS, whole));
    s = s.replace(/_\{([^{}]*)\}/g, (whole: string, group: string) => MarkdownParser.mapExponents(group, SUBSCRIPTS, whole));
    s = s.replace(/\^(\S)/g, (whole: string, group: string) => MarkdownParser.mapExponents(group, SUPERSCRIPTS, whole));
    s = s.replace(/_(\S)/g, (whole: string, group: string) => MarkdownParser.mapExponents(group, SUBSCRIPTS, whole));
    s = s.replace(/\\(left|right|big|Big|bigg|Bigg)/g, '');
    s = s.replace(/\\[,;!]/g, ' ');
    s = s.replace(/\\(quad|qquad)/g, '  ');
    s = s.replace(/\\\\/g, ' ');
    s = s.replace(/\\([a-zA-Z]+)/g, '$1');
    s = s.replace(/[{}]/g, '');
    s = s.replace(/\s+/g, ' ');
    return s.trim();
  }

  private static mapExponents(group: string, table: Record<string, string>, fallback: string): string {
    let out = '';
    for (let index = 0; index < group.length; index++) {
      const char = group[index];
      const mapped = table[char];
      if (mapped === undefined) {
        return fallback;
      }
      out += mapped;
    }
    return out;
  }

  private static replaceInlineMath(text: string): string {
    const render = (whole: string, group: string): string => MarkdownParser.latexToUnicode(group);
    // Display pairs on a single line first, otherwise their outer dollars survive the inline pass.
    return text.replace(/\$\$([^$\n]+)\$\$/g, render).replace(/\$([^$\n]+)\$/g, render);
  }

  static parse(source: string): MarkdownBlock[] {
    const blocks: MarkdownBlock[] = [];
    const lines = source.replace(/\r\n/g, '\n').split('\n');
    let index = 0;
    let blockId = 0;
    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        index++;
        continue;
      }
      if (line.trim().startsWith('```')) {
        const codeLines: string[] = [];
        index++;
        while (index < lines.length && !lines[index].trim().startsWith('```')) {
          codeLines.push(lines[index]);
          index++;
        }
        if (index < lines.length) {
          index++;
        }
        blocks.push(MarkdownParser.createBlock(blockId++, MarkdownBlockType.CODE, codeLines.join('\n')));
        continue;
      }
      if (line.trim() === '$$') {
        const mathLines: string[] = [];
        index++;
        while (index < lines.length && lines[index].trim() !== '$$') {
          mathLines.push(lines[index]);
          index++;
        }
        if (index < lines.length) {
          index++;
        }
        blocks.push(MarkdownParser.createBlock(blockId++, MarkdownBlockType.MATH, mathLines.join('\n')));
        continue;
      }
      if (line.startsWith('#')) {
        blocks.push(MarkdownParser.createBlock(blockId++, MarkdownBlockType.HEADING,
          line.replace(/^#+\s*/, '')));
        index++;
        continue;
      }
      if (line.trim().startsWith('>')) {
        blocks.push(MarkdownParser.createBlock(blockId++, MarkdownBlockType.QUOTE,
          line.trim().replace(/^>\s*/, '')));
        index++;
        continue;
      }
      if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
        const listLines: string[] = [];
        while (index < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[index])) {
          listLines.push('• ' + lines[index].replace(/^\s*([-*]|\d+\.)\s+/, ''));
          index++;
        }
        blocks.push(MarkdownParser.createBlock(blockId++, MarkdownBlockType.LIST, listLines.join('\n')));
        continue;
      }
      if (line.indexOf('|') >= 0 && index + 1 < lines.length && /^\s*\|?\s*:?-+/.test(lines[index + 1])) {
        const tableLines: string[] = [MarkdownParser.formatTableRow(line)];
        index += 2;
        while (index < lines.length && lines[index].indexOf('|') >= 0 && lines[index].trim()) {
          tableLines.push(MarkdownParser.formatTableRow(lines[index]));
          index++;
        }
        blocks.push(MarkdownParser.createBlock(blockId++, MarkdownBlockType.TABLE, tableLines.join('\n')));
        continue;
      }
      const paragraphs: string[] = [line];
      index++;
      while (index < lines.length && lines[index].trim() && !MarkdownParser.startsBlock(lines, index)) {
        paragraphs.push(lines[index]);
        index++;
      }
      blocks.push(MarkdownParser.createBlock(blockId++, MarkdownBlockType.PARAGRAPH, paragraphs.join('\n')));
    }
    return blocks;
  }

  private static startsBlock(lines: string[], index: number): boolean {
    const line = lines[index];
    if (line.trim().startsWith('```') || line.trim() === '$$' || line.startsWith('#') ||
      line.trim().startsWith('>') || /^\s*([-*]|\d+\.)\s+/.test(line)) {
      return true;
    }
    return line.indexOf('|') >= 0 && index + 1 < lines.length && /^\s*\|?\s*:?-+/.test(lines[index + 1]);
  }

  private static createBlock(id: number, type: MarkdownBlockType, content: string): MarkdownBlock {
    const block = new MarkdownBlock();
    block.id = id;
    block.type = type;
    if (type === MarkdownBlockType.MATH) {
      block.content = MarkdownParser.latexToUnicode(content);
    } else if (type === MarkdownBlockType.PARAGRAPH || type === MarkdownBlockType.HEADING ||
      type === MarkdownBlockType.LIST || type === MarkdownBlockType.QUOTE) {
      block.content = MarkdownParser.replaceInlineMath(content);
    } else {
      block.content = content;
    }
    return block;
  }

  private static formatTableRow(line: string): string {
    return line.split('|').map((cell: string) => cell.trim()).filter((cell: string) => cell.length > 0).join('  |  ');
  }
}

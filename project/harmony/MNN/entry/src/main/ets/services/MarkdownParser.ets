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

export class MarkdownParser {
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
    block.content = content;
    return block;
  }

  private static formatTableRow(line: string): string {
    return line.split('|').map((cell: string) => cell.trim()).filter((cell: string) => cell.length > 0).join('  |  ');
  }
}

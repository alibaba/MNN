export class ReasoningResult {
  reasoning: string = '';
  answer: string = '';
  isThinking: boolean = false;
}

export class ReasoningParser {
  static parse(rawText: string): ReasoningResult {
    const result = new ReasoningResult();
    const openTag = '<think>';
    const closeTag = '</think>';
    const openIndex = rawText.indexOf(openTag);
    if (openIndex < 0) {
      result.answer = rawText;
      return result;
    }

    const beforeThinking = rawText.substring(0, openIndex);
    const reasoningStart = openIndex + openTag.length;
    const closeIndex = rawText.indexOf(closeTag, reasoningStart);
    if (closeIndex < 0) {
      result.reasoning = rawText.substring(reasoningStart).trim();
      result.answer = beforeThinking.trim();
      result.isThinking = true;
      return result;
    }

    result.reasoning = rawText.substring(reasoningStart, closeIndex).trim();
    const afterThinking = rawText.substring(closeIndex + closeTag.length);
    result.answer = `${beforeThinking}${afterThinking}`.trim();
    return result;
  }
}

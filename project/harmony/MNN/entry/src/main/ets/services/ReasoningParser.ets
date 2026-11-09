export class ReasoningResult {
  reasoning: string = '';
  answer: string = '';
  isThinking: boolean = false;
}

/**
 * Splits a reply into its thinking section and the user-facing answer.
 *
 * This follows the Android client's GenerateResultProcessor (THINK_TAGS format) rather than only
 * looking for an opening ` thinking` tag: the chat template usually puts ` thinking` in the prompt,
 * so the model only emits the closing tag. In that case everything up to the closing tag is
 * reasoning and the rest is the answer. While the closing tag has not arrived yet the text is shown
 * as normal output, exactly like the Android processor does with its pending buffer.
 */
export class ReasoningParser {
  private static readonly OPEN_TAG: string = ' thinking';
  private static readonly CLOSE_TAG: string = '\u003c/think\u003e';

  static parse(rawText: string): ReasoningResult {
    const result = new ReasoningResult();
    if (!rawText) {
      return result;
    }
    const openTag = ReasoningParser.OPEN_TAG;
    const closeTag = ReasoningParser.CLOSE_TAG;

    const closeIndex = rawText.lastIndexOf(closeTag);
    if (closeIndex < 0) {
      const openIndex = rawText.lastIndexOf(openTag);
      if (openIndex < 0) {
        result.answer = rawText;
        return result;
      }
      result.reasoning = rawText.substring(openIndex + openTag.length).trim();
      result.answer = rawText.substring(0, openIndex).trim();
      result.isThinking = true;
      return result;
    }

    const beforeThinking = rawText.substring(0, closeIndex);
    const afterThinking = rawText.substring(closeIndex + closeTag.length);
    const openIndex = beforeThinking.lastIndexOf(openTag);
    if (openIndex >= 0) {
      result.reasoning = beforeThinking.substring(openIndex + openTag.length).trim();
      result.answer = `${beforeThinking.substring(0, openIndex)}${afterThinking}`.trim();
    } else {
      // Qwen3-style output: the opening tag only exists in the prompt.
      result.reasoning = beforeThinking.trim();
      result.answer = afterThinking.trim();
    }
    return result;
  }
}
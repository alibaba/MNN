export interface LlmLoadResult {
  loadMs: number;
  reused: boolean;
}

export interface LlmChatMessage {
  role: string;
  content: string;
}

export interface LlmGenerateRequest {
  requestId: string;
  messages: LlmChatMessage[];
  maxTokens: number;
}

export interface LlmGenerateResult {
  requestId: string;
  text: string;
  promptTokens: number;
  generatedTokens: number;
  prefillMs: number;
  decodeMs: number;
  totalMs: number;
  firstTokenMs: number;
  canceled: boolean;
}

export interface LlmBenchmarkRequest {
  requestId: string;
  promptTokens: number;
  generateTokens: number;
  repeats: number;
}

export interface LlmBenchmarkSample {
  iteration: number;
  promptTokens: number;
  generatedTokens: number;
  prefillMs: number;
  decodeMs: number;
  totalMs: number;
}

export interface LlmBenchmarkResult {
  requestId: string;
  samples: LlmBenchmarkSample[];
  canceled: boolean;
}

export const loadLlm: (configPath: string, runtimeConfigJson: string) => Promise<LlmLoadResult>;
export const generateLlm: (
  request: LlmGenerateRequest,
  onChunk: (chunk: string) => void
) => Promise<LlmGenerateResult>;
export const runLlmBenchmark: (
  request: LlmBenchmarkRequest,
  onProgress: (sample: LlmBenchmarkSample) => void
) => Promise<LlmBenchmarkResult>;
export const cancelLlm: (requestId: string) => void;
export const resetLlm: () => void;
export const releaseLlm: () => void;
export const getMnnVersion: () => string;

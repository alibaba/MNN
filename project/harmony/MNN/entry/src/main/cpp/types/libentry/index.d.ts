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
  audioOutputPath?: string;
  streamAudio?: boolean;
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

export interface LlmDiffusionRequest {
  requestId: string;
  resourcePath: string;
  prompt: string;
  outputPath: string;
  modelType: number;
  backendType: number;
  memoryMode: number;
  width: number;
  height: number;
  steps: number;
  seed: number;
  useCfg?: boolean;
  cfgScale?: number;
  negativePrompt?: string;
  imageInput?: string;
}

export interface LlmDiffusionResult {
  requestId: string;
  success: boolean;
  totalMs: number;
}

export const loadLlm: (configPath: string, runtimeConfigJson: string) => Promise<LlmLoadResult>;
export const generateLlm: (
  request: LlmGenerateRequest,
  onChunk: (chunk: string) => void,
  onAudio?: (pcm: Int16Array) => void
) => Promise<LlmGenerateResult>;
export const runLlmBenchmark: (
  request: LlmBenchmarkRequest,
  onProgress: (sample: LlmBenchmarkSample) => void
) => Promise<LlmBenchmarkResult>;
export const cancelLlm: (requestId: string) => void;
export const resetLlm: () => void;
export const releaseLlm: () => void;
export const getMnnVersion: () => string;
export const dumpLlmConfig: () => string;
export const runDiffusion: (
  request: LlmDiffusionRequest,
  onProgress: (progress: number) => void
) => Promise<LlmDiffusionResult>;

export interface NativeBenchResult {
  exitCode: number;
  output: string;
}

// Runs llm_bench in-process with the given command-line string (e.g.
// "-m /path/config.json -a cpu -t 4 -c 2 -pg 512,128 -p 0 -n 0 -rep 1") and
// resolves with its captured stdout/stderr. One run at a time; rejects if busy.
export const runNativeBench: (args: string) => Promise<NativeBenchResult>;

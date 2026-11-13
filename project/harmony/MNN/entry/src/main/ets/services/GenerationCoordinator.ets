import testNapi from 'libentry.so';
import { MemoryProbe } from './MemoryProbe';

export interface NativeTarget {
  modelId: string;
  configPath: string;
  runtimeConfigJson: string;
}

export interface NativeLoadResult {
  loadMs: number;
  reused: boolean;
}

export interface NativeChatMessage {
  role: string;
  content: string;
}

export interface NativeGenerateRequest {
  requestId: string;
  messages: NativeChatMessage[];
  maxTokens: number;
  audioOutputPath?: string;
}

export interface NativeGenerateResult {
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

export interface NativeBenchmarkRequest {
  requestId: string;
  promptTokens: number;
  generateTokens: number;
  repeats: number;
}

export interface NativeBenchmarkSample {
  iteration: number;
  promptTokens: number;
  generatedTokens: number;
  prefillMs: number;
  decodeMs: number;
  totalMs: number;
}

export interface NativeBenchmarkResult {
  requestId: string;
  samples: NativeBenchmarkSample[];
  canceled: boolean;
}

export function mnnVersion(): string {
  try {
    return testNapi.getMnnVersion() as string;
  } catch (error) {
    return '';
  }
}

export function nativeLlmConfig(): string {
  try {
    return testNapi.dumpLlmConfig() as string;
  } catch (error) {
    return '';
  }
}

export class QueueStats {
  totalRequests: number = 0;
  completedRequests: number = 0;
  failedRequests: number = 0;
  pendingRequests: number = 0;
  isProcessing: boolean = false;
  successRate: number = 0;
}

export class GenerationCoordinator {
  private static instance: GenerationCoordinator = new GenerationCoordinator();
  private tail: Promise<void> = Promise.resolve();
  private canceled: Set<string> = new Set<string>();
  private pendingIds: Set<string> = new Set<string>();
  private activeRequestId: string = '';
  private sequence: number = 0;
  private waiting: number = 0;
  private totalRequests: number = 0;
  private completedRequests: number = 0;
  private failedRequests: number = 0;

  static getInstance(): GenerationCoordinator {
    return GenerationCoordinator.instance;
  }

  createRequestId(owner: string): string {
    return `${owner}:${Date.now()}:${++this.sequence}`;
  }

  queueSize(): number {
    return this.waiting + (this.activeRequestId ? 1 : 0);
  }

  getQueueStats(): QueueStats {
    const stats = new QueueStats();
    stats.totalRequests = this.totalRequests;
    stats.completedRequests = this.completedRequests;
    stats.failedRequests = this.failedRequests;
    stats.pendingRequests = this.pendingIds.size;
    stats.isProcessing = this.activeRequestId.length > 0;
    stats.successRate = this.totalRequests > 0 ? this.completedRequests * 100 / this.totalRequests : 0;
    return stats;
  }

  /** Cancels queued (not yet running) requests and returns how many were dropped. */
  clearPendingQueue(): number {
    let cleared = 0;
    this.pendingIds.forEach((requestId: string) => {
      this.canceled.add(requestId);
      cleared++;
    });
    this.pendingIds.clear();
    return cleared;
  }

  prepare(target: NativeTarget): Promise<NativeLoadResult> {
    return this.enqueue(async () => {
      const result = await testNapi.loadLlm(target.configPath, target.runtimeConfigJson) as NativeLoadResult;
      MemoryProbe.snapshot('model-loaded', target.runtimeConfigJson);
      return result;
    });
  }

  generate(target: NativeTarget, request: NativeGenerateRequest,
    onChunk: (chunk: string) => void): Promise<NativeGenerateResult> {
    this.totalRequests++;
    this.pendingIds.add(request.requestId);
    const pending = this.enqueue(async () => {
      this.pendingIds.delete(request.requestId);
      if (this.canceled.delete(request.requestId)) {
        return this.canceledGenerateResult(request.requestId);
      }
      await testNapi.loadLlm(target.configPath, target.runtimeConfigJson);
      MemoryProbe.snapshot('gen-loaded', target.runtimeConfigJson);
      this.activeRequestId = request.requestId;
      try {
        const result = await testNapi.generateLlm(request, onChunk) as NativeGenerateResult;
        MemoryProbe.snapshot('gen-done', target.runtimeConfigJson);
        return result;
      } finally {
        this.activeRequestId = '';
      }
    });
    return pending.then((result: NativeGenerateResult) => {
      this.completedRequests++;
      return result;
    }, (error: Error) => {
      this.failedRequests++;
      throw error;
    });
  }

  benchmark(target: NativeTarget, request: NativeBenchmarkRequest,
    onProgress: (sample: NativeBenchmarkSample) => void): Promise<NativeBenchmarkResult> {
    this.totalRequests++;
    this.pendingIds.add(request.requestId);
    const pending = this.enqueue(async () => {
      this.pendingIds.delete(request.requestId);
      if (this.canceled.delete(request.requestId)) {
        return this.canceledBenchmarkResult(request.requestId);
      }
      await testNapi.loadLlm(target.configPath, target.runtimeConfigJson);
      this.activeRequestId = request.requestId;
      try {
        return await testNapi.runLlmBenchmark(request, onProgress) as NativeBenchmarkResult;
      } finally {
        this.activeRequestId = '';
      }
    });
    return pending.then((result: NativeBenchmarkResult) => {
      this.completedRequests++;
      return result;
    }, (error: Error) => {
      this.failedRequests++;
      throw error;
    });
  }

  cancel(requestId: string): void {
    this.canceled.add(requestId);
    if (this.activeRequestId === requestId) {
      testNapi.cancelLlm(requestId);
    }
  }

  shutdown(): Promise<void> {
    if (this.activeRequestId) {
      this.cancel(this.activeRequestId);
    }
    return this.enqueue(async () => {
      testNapi.releaseLlm();
    });
  }

  private enqueue<T>(action: () => Promise<T>): Promise<T> {
    this.waiting++;
    const result = this.tail.then(async () => {
      this.waiting--;
      return await action();
    });
    this.tail = result.then(() => undefined, () => undefined);
    return result;
  }

  private canceledGenerateResult(requestId: string): NativeGenerateResult {
    return {
      requestId: requestId,
      text: '',
      promptTokens: 0,
      generatedTokens: 0,
      prefillMs: 0,
      decodeMs: 0,
      totalMs: 0,
      firstTokenMs: 0,
      canceled: true
    };
  }

  private canceledBenchmarkResult(requestId: string): NativeBenchmarkResult {
    return {
      requestId: requestId,
      samples: [],
      canceled: true
    };
  }
}

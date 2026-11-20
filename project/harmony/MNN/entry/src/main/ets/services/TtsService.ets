import { textToSpeech } from '@kit.CoreSpeechKit';

/**
 * System TTS used by continuous voice chat when the loaded model has no talker head,
 * mirroring the Android client's reply playback for text-only models.
 */
export class TtsService {
  private static instance: TtsService = new TtsService();
  private engine?: textToSpeech.TextToSpeechEngine;
  private pendingDone: (() => void) | null = null;
  private currentRequestId: string = '';
  private seq: number = 0;

  static getInstance(): TtsService {
    return TtsService.instance;
  }

  isSpeaking(): boolean {
    return this.pendingDone !== null;
  }

  private createEngine(): Promise<textToSpeech.TextToSpeechEngine> {
    return new Promise<textToSpeech.TextToSpeechEngine>((resolve, reject) => {
      textToSpeech.createEngine({ language: 'zh-CN', person: 0, online: 1 },
        (error, engine) => {
          if (error || !engine) {
            reject(new Error(`TTS engine create failed: ${error ? error.message : 'unknown'}`));
          } else {
            resolve(engine);
          }
        });
    });
  }

  async speak(text: string, onDone: () => void): Promise<void> {
    if (!this.engine) {
      this.engine = await this.createEngine();
      this.engine.setListener({
        onStart: (requestId: string, response: textToSpeech.StartResponse) => {
        },
        onComplete: (requestId: string, response: textToSpeech.CompleteResponse) => {
          this.finishFor(requestId);
        },
        onStop: (requestId: string, response: textToSpeech.StopResponse) => {
          this.finishFor(requestId);
        },
        onError: (requestId: string, errorCode: number, errorMessage: string) => {
          this.finishFor(requestId);
        }
      });
    }
    const engine = this.engine;
    if (!engine) {
      onDone();
      return;
    }
    if (this.pendingDone) {
      // Never strand a waiter (the voice-chat loop pauses the mic until onDone fires).
      const stranded = this.pendingDone;
      this.pendingDone = null;
      this.currentRequestId = '';
      try {
        engine.stop();
      } catch (error) {
      }
      stranded();
    }
    this.pendingDone = onDone;
    this.seq++;
    this.currentRequestId = `tts_${this.seq}`;
    const params: textToSpeech.SpeakParams = { requestId: this.currentRequestId };
    engine.speak(text, params);
  }

  stop(): void {
    this.currentRequestId = '';
    const engine = this.engine;
    if (engine) {
      try {
        engine.stop();
      } catch (error) {
      }
    }
    this.finish();
  }

  /** Async engine callbacks from a superseded speak must not release the current waiter. */
  private finishFor(requestId: string): void {
    if (this.currentRequestId !== '' && requestId !== this.currentRequestId) {
      return;
    }
    this.finish();
  }

  private finish(): void {
    const done = this.pendingDone;
    this.pendingDone = null;
    if (done) {
      done();
    }
  }
}

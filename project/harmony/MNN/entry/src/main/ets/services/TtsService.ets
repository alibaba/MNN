import { textToSpeech } from '@kit.CoreSpeechKit';

/**
 * System TTS used by continuous voice chat when the loaded model has no talker head,
 * mirroring the Android client's reply playback for text-only models.
 */
export class TtsService {
  private static instance: TtsService = new TtsService();
  private engine?: textToSpeech.TextToSpeechEngine;
  private pendingDone: (() => void) | null = null;

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
          this.finish();
        },
        onStop: (requestId: string, response: textToSpeech.StopResponse) => {
          this.finish();
        },
        onError: (requestId: string, errorCode: number, errorMessage: string) => {
          this.finish();
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
      try {
        engine.stop();
      } catch (error) {
      }
      stranded();
    }
    this.pendingDone = onDone;
    const params: textToSpeech.SpeakParams = { requestId: `tts_${Date.now()}` };
    engine.speak(text, params);
  }

  stop(): void {
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

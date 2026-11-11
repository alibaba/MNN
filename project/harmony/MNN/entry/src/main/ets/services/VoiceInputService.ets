import { tr } from './I18n';
import { abilityAccessCtrl, common, Permissions } from '@kit.AbilityKit';
import { audio } from '@kit.AudioKit';
import { speechRecognizer } from '@kit.CoreSpeechKit';

const MICROPHONE_PERMISSION: Permissions = 'ohos.permission.MICROPHONE';
const SAMPLE_RATE: number = 16000;
// The offline recognizer accepts exactly 640 or 1280 bytes per writeAudio call (20/40 ms).
const CHUNK_BYTES: number = 1280;

export class VoiceInputCallbacks {
  onPartial: (text: string) => void = () => {
  };
  onFinal: (text: string) => void = () => {
  };
  onState: (listening: boolean, message: string) => void = () => {
  };
}

/**
 * Streaming voice input built on the on-device speech recognizer (Chinese, offline).
 * Captures 16 kHz mono PCM and feeds fixed 40 ms chunks while partial results stream back,
 * mirroring the Android client's voice chat input.
 */
export class VoiceInputService {
  private static instance: VoiceInputService = new VoiceInputService();
  private capturer?: audio.AudioCapturer;
  private engine?: speechRecognizer.SpeechRecognitionEngine;
  private sessionId: string = '';
  private pending: Uint8Array = new Uint8Array(0);
  private listening: boolean = false;
  private finishTimerId: number = -1;
  private callbacks: VoiceInputCallbacks = new VoiceInputCallbacks();
  private readCallback: (buffer: ArrayBuffer) => void = (buffer: ArrayBuffer) => this.feed(buffer);

  static getInstance(): VoiceInputService {
    return VoiceInputService.instance;
  }

  isListening(): boolean {
    return this.listening;
  }

  async start(context: common.UIAbilityContext, callbacks: VoiceInputCallbacks): Promise<void> {
    if (this.listening) {
      return;
    }
    this.clearFinishTimer();
    this.callbacks = callbacks;
    const permissionResult = await abilityAccessCtrl.createAtManager()
      .requestPermissionsFromUser(context, [MICROPHONE_PERMISSION]);
    if (permissionResult.authResults.length === 0 || permissionResult.authResults[0] !== 0) {
      throw new Error(tr('麦克风权限未授予'));
    }
    const engine = await speechRecognizer.createEngine({ language: 'zh-CN', online: 1 });
    engine.setListener({
      onStart: (sessionId: string, eventMessage: string) => {
      },
      onEvent: (sessionId: string, eventCode: number, eventMessage: string) => {
      },
      onResult: (sessionId: string, result: speechRecognizer.SpeechRecognitionResult) => {
        if (result.isFinal) {
          this.callbacks.onFinal(result.result);
        } else {
          this.callbacks.onPartial(result.result);
        }
      },
      onComplete: (sessionId: string, eventMessage: string) => {
        console.info(`[VoiceInput] complete: ${eventMessage}`);
        this.clearFinishTimer();
        this.releaseCapture();
        this.releaseEngine();
        this.callbacks.onState(false, '');
      },
      onError: (sessionId: string, errorCode: number, errorMessage: string) => {
        console.error(`[VoiceInput] error ${errorCode}: ${errorMessage}`);
        this.clearFinishTimer();
        this.releaseCapture();
        this.releaseEngine();
        this.listening = false;
        this.callbacks.onState(false, `${tr('语音识别失败(')}${errorCode})：${errorMessage}`);
      }
    });
    this.engine = engine;
    this.sessionId = `voice_${Date.now()}`;
    engine.startListening({
      sessionId: this.sessionId,
      audioInfo: {
        audioType: 'pcm',
        sampleRate: SAMPLE_RATE,
        soundChannel: 1,
        sampleBit: 16
      }
    });

    const options: audio.AudioCapturerOptions = {
      streamInfo: {
        samplingRate: audio.AudioSamplingRate.SAMPLE_RATE_16000,
        channels: audio.AudioChannel.CHANNEL_1,
        sampleFormat: audio.AudioSampleFormat.SAMPLE_FORMAT_S16LE,
        encodingType: audio.AudioEncodingType.ENCODING_TYPE_RAW
      },
      capturerInfo: {
        source: audio.SourceType.SOURCE_TYPE_VOICE_RECOGNITION,
        capturerFlags: 0
      }
    };
    const capturer = await audio.createAudioCapturer(options);
    try {
      capturer.on('readData', this.readCallback);
      await capturer.start();
      this.capturer = capturer;
      this.pending = new Uint8Array(0);
      this.listening = true;
      this.callbacks.onState(true, tr('正在听…'));
    } catch (error) {
      capturer.off('readData', this.readCallback);
      await capturer.release();
      this.cancel();
      throw error as Error;
    }
  }

  /** Stops capture and asks the recognizer to flush the final result. */
  async stop(): Promise<void> {
    if (!this.listening) {
      return;
    }
    this.listening = false;
    this.pending = new Uint8Array(0);
    const engine = this.engine;
    if (engine) {
      try {
        engine.finish(this.sessionId);
        this.callbacks.onState(false, tr('正在识别…'));
        // Guard against a missing onComplete (for example when no speech was detected).
        this.finishTimerId = setTimeout(() => {
          this.finishTimerId = -1;
          this.releaseEngine();
          this.callbacks.onState(false, '');
        }, 6000);
      } catch (error) {
        this.releaseEngine();
        this.callbacks.onState(false, '');
      }
    } else {
      this.callbacks.onState(false, '');
    }
    // Stopping the capturer can take a moment; the UI already reflects the stopped state.
    await this.releaseCapture();
  }

  /** Aborts recognition without waiting for a final result. */
  async cancel(): Promise<void> {
    this.listening = false;
    this.clearFinishTimer();
    await this.releaseCapture();
    this.pending = new Uint8Array(0);
    const engine = this.engine;
    if (engine) {
      try {
        engine.cancel(this.sessionId);
      } catch (error) {
      }
    }
    this.releaseEngine();
  }

  private feed(buffer: ArrayBuffer): void {
    const engine = this.engine;
    if (!this.listening || !engine) {
      return;
    }
    const incoming = new Uint8Array(buffer);
    const merged = new Uint8Array(this.pending.length + incoming.length);
    merged.set(this.pending, 0);
    merged.set(incoming, this.pending.length);
    let offset = 0;
    while (merged.length - offset >= CHUNK_BYTES) {
      try {
        engine.writeAudio(this.sessionId, merged.slice(offset, offset + CHUNK_BYTES));
      } catch (error) {
        this.stop();
        return;
      }
      offset += CHUNK_BYTES;
    }
    this.pending = merged.slice(offset);
  }

  private clearFinishTimer(): void {
    if (this.finishTimerId >= 0) {
      clearTimeout(this.finishTimerId);
      this.finishTimerId = -1;
    }
  }

  private async releaseCapture(): Promise<void> {
    const capturer = this.capturer;
    this.capturer = undefined;
    if (!capturer) {
      return;
    }
    try {
      await capturer.stop();
    } catch (error) {
    }
    try {
      capturer.off('readData', this.readCallback);
      await capturer.release();
    } catch (error) {
    }
  }

  private releaseEngine(): void {
    const engine = this.engine;
    this.engine = undefined;
    if (!engine) {
      return;
    }
    try {
      engine.shutdown();
    } catch (error) {
    }
  }
}
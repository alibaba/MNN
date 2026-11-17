import { tr } from './I18n';
import { abilityAccessCtrl, common, Permissions } from '@kit.AbilityKit';
import { audio } from '@kit.AudioKit';
import { speechRecognizer } from '@kit.CoreSpeechKit';

const MICROPHONE_PERMISSION: Permissions = 'ohos.permission.MICROPHONE';
const SAMPLE_RATE: number = 16000;
const CHUNK_BYTES: number = 1280;
// Energy VAD thresholds over int16 PCM RMS; tuned for a handset mic in a quiet room.
const START_RMS: number = 420;
const STOP_RMS: number = 280;
const SILENCE_MS: number = 900;
const MIN_SPEECH_MS: number = 250;
const MAX_UTTERANCE_MS: number = 20000;

export class VoiceChatCallbacks {
  /** state: 'listening' | 'speech' | 'recognizing' | 'paused' */
  onState: (state: string, partial: string) => void = () => {
  };
  onUtterance: (text: string) => void = () => {
  };
  onError: (message: string) => void = () => {
  };
}

/**
 * Hands-free continuous voice conversation: keeps the microphone open, cuts utterances with an
 * energy VAD, streams them through the on-device recognizer and hands finals to the chat loop.
 * Mirrors the Android client's sherpa streaming ASR + VAD voice-chat mode.
 */
export class VoiceChatService {
  private static instance: VoiceChatService = new VoiceChatService();
  private capturer?: audio.AudioCapturer;
  private engine?: speechRecognizer.SpeechRecognitionEngine;
  private sessionId: string = '';
  private pending: Uint8Array = new Uint8Array(0);
  private callbacks: VoiceChatCallbacks = new VoiceChatCallbacks();
  private running: boolean = false;
  private paused: boolean = false;
  private phase: string = 'idle';
  private speechStart: number = 0;
  private lastLoud: number = 0;
  private partial: string = '';
  private consecutiveErrors: number = 0;
  private readCallback: (buffer: ArrayBuffer) => void = (buffer: ArrayBuffer) => this.feed(buffer);

  static getInstance(): VoiceChatService {
    return VoiceChatService.instance;
  }

  isRunning(): boolean {
    return this.running;
  }

  /** Gates the mic while the model talks back, so the reply is not transcribed as user speech. */
  setPaused(paused: boolean): void {
    if (this.paused === paused) {
      return;
    }
    this.paused = paused;
    if (paused) {
      this.abortUtterance();
      this.phase = 'idle';
      this.callbacks.onState('paused', '');
    } else if (this.running) {
      this.phase = 'listening';
      this.callbacks.onState('listening', '');
    }
  }

  async start(context: common.UIAbilityContext, callbacks: VoiceChatCallbacks): Promise<void> {
    if (this.running) {
      return;
    }
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
        this.consecutiveErrors = 0;
        if (result.isFinal) {
          const text = result.result.trim();
          this.partial = '';
          this.phase = 'listening';
          this.callbacks.onState('listening', '');
          if (text && this.running) {
            this.callbacks.onUtterance(text);
          }
        } else {
          this.partial = result.result;
          this.callbacks.onState('speech', this.partial);
        }
      },
      onComplete: (sessionId: string, eventMessage: string) => {
        // Re-arm a fresh session so the conversation continues hands-free.
        this.partial = '';
        if (this.running && !this.paused) {
          this.openSession();
          this.phase = 'listening';
          this.callbacks.onState('listening', '');
        }
      },
      onError: (sessionId: string, errorCode: number, errorMessage: string) => {
        console.error(`[VoiceChat] error ${errorCode}: ${errorMessage}`);
        this.partial = '';
        this.consecutiveErrors++;
        if (!this.running) {
          return;
        }
        if (this.consecutiveErrors >= 3) {
          // A persistently failing engine would otherwise reopen sessions forever.
          this.stop().catch(() => {
          });
          this.callbacks.onError(`${tr('语音识别失败(')}${errorCode})：${errorMessage}`);
          return;
        }
        this.openSession();
        this.phase = 'listening';
      }
    });
    this.engine = engine;
    this.running = true;
    this.paused = false;
    this.openSession();

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
    let capturer: audio.AudioCapturer;
    try {
      capturer = await audio.createAudioCapturer(options);
    } catch (error) {
      this.running = false;
      this.shutdownEngine();
      throw error as Error;
    }
    try {
      capturer.on('readData', this.readCallback);
      await capturer.start();
      this.capturer = capturer;
      this.phase = 'listening';
      this.callbacks.onState('listening', '');
    } catch (error) {
      capturer.off('readData', this.readCallback);
      await capturer.release();
      this.running = false;
      this.shutdownEngine();
      throw error as Error;
    }
  }

  async stop(): Promise<void> {
    this.running = false;
    this.paused = false;
    this.phase = 'idle';
    this.partial = '';
    const capturer = this.capturer;
    this.capturer = undefined;
    if (capturer) {
      try {
        await capturer.stop();
        capturer.off('readData', this.readCallback);
        await capturer.release();
      } catch (error) {
      }
    }
    const engine = this.engine;
    this.engine = undefined;
    if (engine) {
      try {
        engine.cancel(this.sessionId);
      } catch (error) {
      }
      try {
        engine.shutdown();
      } catch (error) {
      }
    }
    this.callbacks.onState('idle', '');
  }

  private shutdownEngine(): void {
    const engine = this.engine;
    this.engine = undefined;
    if (engine) {
      try {
        engine.shutdown();
      } catch (error) {
      }
    }
  }

  private openSession(): void {
    const engine = this.engine;
    if (!engine) {
      return;
    }
    this.sessionId = `voicechat_${Date.now()}`;
    this.pending = new Uint8Array(0);
    try {
      engine.startListening({
        sessionId: this.sessionId,
        audioInfo: {
          audioType: 'pcm',
          sampleRate: SAMPLE_RATE,
          soundChannel: 1,
          sampleBit: 16
        }
      });
    } catch (error) {
      this.callbacks.onError((error as Error).message);
    }
  }

  private abortUtterance(): void {
    const engine = this.engine;
    if (!engine || this.phase !== 'speech') {
      return;
    }
    try {
      engine.cancel(this.sessionId);
    } catch (error) {
    }
    this.openSession();
  }

  private feed(buffer: ArrayBuffer): void {
    if (!this.running || this.paused || !this.engine) {
      return;
    }
    const incoming = new Uint8Array(buffer);
    const rms = VoiceChatService.rmsOf(incoming);
    const now = Date.now();
    if (this.phase === 'listening') {
      if (rms > START_RMS) {
        this.phase = 'speech';
        this.speechStart = now;
        this.lastLoud = now;
        this.callbacks.onState('speech', '');
      } else {
        return;
      }
    } else if (this.phase !== 'speech') {
      return;
    }
    if (rms > STOP_RMS) {
      this.lastLoud = now;
    }
    const merged = new Uint8Array(this.pending.length + incoming.length);
    merged.set(this.pending, 0);
    merged.set(incoming, this.pending.length);
    let offset = 0;
    while (merged.length - offset >= CHUNK_BYTES) {
      try {
        this.engine.writeAudio(this.sessionId, merged.slice(offset, offset + CHUNK_BYTES));
      } catch (error) {
        this.openSession();
        this.phase = 'listening';
        return;
      }
      offset += CHUNK_BYTES;
    }
    this.pending = merged.slice(offset);
    if (now - this.speechStart > MAX_UTTERANCE_MS ||
      (now - this.lastLoud > SILENCE_MS && now - this.speechStart > MIN_SPEECH_MS)) {
      this.phase = 'recognizing';
      this.callbacks.onState('recognizing', this.partial);
      try {
        this.engine.finish(this.sessionId);
      } catch (error) {
        this.openSession();
        this.phase = 'listening';
      }
    }
  }

  private static rmsOf(bytes: Uint8Array): number {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const samples = Math.floor(bytes.byteLength / 2);
    if (samples === 0) {
      return 0;
    }
    let sum = 0;
    for (let index = 0; index < samples; index++) {
      const sample = view.getInt16(index * 2, true);
      sum += sample * sample;
    }
    return Math.sqrt(sum / samples);
  }
}

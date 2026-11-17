import { media } from '@kit.MediaKit';
import { audio } from '@kit.AudioKit';
import { fileIo } from '@kit.CoreFileKit';

/** Minimal AVPlayer wrapper used to replay generated or recorded speech files. */
export class AudioPlaybackService {
  private static instance: AudioPlaybackService = new AudioPlaybackService();
  private player?: media.AVPlayer;
  private file?: fileIo.File;
  private renderer?: audio.AudioRenderer;
  private rendererReady: Promise<void> = Promise.resolve();

  static getInstance(): AudioPlaybackService {
    return AudioPlaybackService.instance;
  }

  async play(path: string): Promise<void> {
    await this.stop();
    // AVPlayer only accepts fd:// or network sources, so hand it an open descriptor.
    const file = fileIo.openSync(path, fileIo.OpenMode.READ_ONLY);
    const size = fileIo.statSync(path).size;
    const player = await media.createAVPlayer();
    this.player = player;
    this.file = file;
    player.on('stateChange', (state: media.AVPlayerState) => {
      if (state === 'initialized') {
        player.prepare().catch(() => {
        });
      } else if (state === 'prepared') {
        player.play().catch(() => {
        });
      } else if (state === 'completed') {
        this.releasePlayer(player);
      }
    });
    player.on('error', () => {
      this.releasePlayer(player);
    });
    player.fdSrc = { fd: file.fd, offset: 0, length: size };
  }

  async stop(): Promise<void> {
    const player = this.player;
    this.player = undefined;
    if (player) {
      try {
        await player.release();
      } catch (error) {
      }
    }
    this.closeFile();
  }

  private releasePlayer(player: media.AVPlayer): void {
    player.release().catch(() => {
    });
    if (this.player === player) {
      this.player = undefined;
      this.closeFile();
    }
  }

  private closeFile(): void {
    if (this.file) {
      try {
        fileIo.closeSync(this.file);
      } catch (error) {
      }
      this.file = undefined;
    }
  }

  /**
   * Low-latency PCM sink for realtime Omni speech: the talker waveform chunks arrive while the
   * model is still generating, mirroring Android's stream-while-decoding playback.
   */
  async startStream(): Promise<void> {
    await this.stopStream();
    this.rendererReady = this.createRenderer();
    await this.rendererReady;
  }

  async writeStream(pcm: Int16Array): Promise<void> {
    await this.rendererReady;
    const renderer = this.renderer;
    if (!renderer || pcm.length === 0) {
      return;
    }
    await renderer.write(pcm.buffer as ArrayBuffer);
  }

  async stopStream(): Promise<void> {
    this.rendererReady = Promise.resolve();
    const renderer = this.renderer;
    this.renderer = undefined;
    if (renderer) {
      try {
        await renderer.stop();
        await renderer.release();
      } catch (error) {
      }
    }
  }

  private async createRenderer(): Promise<void> {
    const streamInfo: audio.AudioStreamInfo = {
      samplingRate: audio.AudioSamplingRate.SAMPLE_RATE_24000,
      channels: audio.AudioChannel.CHANNEL_1,
      sampleFormat: audio.AudioSampleFormat.SAMPLE_FORMAT_S16LE,
      encodingType: audio.AudioEncodingType.ENCODING_TYPE_RAW
    };
    const rendererInfo: audio.AudioRendererInfo = {
      usage: audio.StreamUsage.STREAM_USAGE_MUSIC,
      rendererFlags: 0
    };
    const renderer = await audio.createAudioRenderer({ streamInfo: streamInfo, rendererInfo: rendererInfo });
    this.renderer = renderer;
    await renderer.start();
  }
}
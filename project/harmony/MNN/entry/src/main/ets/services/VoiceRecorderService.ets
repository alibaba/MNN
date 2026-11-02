import { abilityAccessCtrl, common, Permissions } from '@kit.AbilityKit';
import { audio } from '@kit.AudioKit';
import { fileIo } from '@kit.CoreFileKit';
import { AttachmentRecord } from '../domain/AppTypes';

const MICROPHONE_PERMISSION: Permissions = 'ohos.permission.MICROPHONE';
const SAMPLE_RATE: number = 16000;
const CHANNELS: number = 1;
const BITS_PER_SAMPLE: number = 16;

export class VoiceRecorderService {
  private static instance: VoiceRecorderService = new VoiceRecorderService();
  private capturer?: audio.AudioCapturer;
  private pcmFile?: fileIo.File;
  private pcmPath: string = '';
  private wavPath: string = '';
  private dataBytes: number = 0;
  private recording: boolean = false;
  private readCallback: (buffer: ArrayBuffer) => void = (buffer: ArrayBuffer) => {
    if (!this.recording || !this.pcmFile) {
      return;
    }
    this.dataBytes += fileIo.writeSync(this.pcmFile.fd, buffer);
  };

  static getInstance(): VoiceRecorderService {
    return VoiceRecorderService.instance;
  }

  async start(context: common.UIAbilityContext, sessionId: string): Promise<void> {
    if (this.capturer) {
      return;
    }
    const permissionResult = await abilityAccessCtrl.createAtManager()
      .requestPermissionsFromUser(context, [MICROPHONE_PERMISSION]);
    if (permissionResult.authResults.length === 0 || permissionResult.authResults[0] !== 0) {
      throw new Error('麦克风权限未授予');
    }

    const directory = `${context.filesDir}/history/${this.safeName(sessionId)}`;
    if (!fileIo.accessSync(directory)) {
      fileIo.mkdirSync(directory, true);
    }
    const baseName = `audio-${Date.now()}-record`;
    this.pcmPath = `${directory}/${baseName}.pcm`;
    this.wavPath = `${directory}/${baseName}.wav`;
    this.pcmFile = fileIo.openSync(this.pcmPath,
      fileIo.OpenMode.CREATE | fileIo.OpenMode.WRITE_ONLY | fileIo.OpenMode.TRUNC);
    this.dataBytes = 0;

    const options: audio.AudioCapturerOptions = {
      streamInfo: {
        samplingRate: audio.AudioSamplingRate.SAMPLE_RATE_16000,
        channels: audio.AudioChannel.CHANNEL_1,
        sampleFormat: audio.AudioSampleFormat.SAMPLE_FORMAT_S16LE,
        encodingType: audio.AudioEncodingType.ENCODING_TYPE_RAW
      },
      capturerInfo: {
        source: audio.SourceType.SOURCE_TYPE_VOICE_MESSAGE,
        capturerFlags: 0
      }
    };
    const capturer = await audio.createAudioCapturer(options);
    try {
      capturer.on('readData', this.readCallback);
      this.recording = true;
      await capturer.start();
      this.capturer = capturer;
    } catch (error) {
      this.recording = false;
      capturer.off('readData', this.readCallback);
      await capturer.release();
      this.closePcm(true);
      throw error as Error;
    }
  }

  async stop(): Promise<AttachmentRecord> {
    if (!this.capturer) {
      throw new Error('当前没有录音任务');
    }
    const capturer = this.capturer;
    this.capturer = undefined;
    this.recording = false;
    await capturer.stop();
    capturer.off('readData', this.readCallback);
    await capturer.release();
    this.closePcm(false);
    this.writeWav();
    fileIo.unlinkSync(this.pcmPath);
    this.pcmPath = '';

    const attachment = new AttachmentRecord();
    attachment.mediaType = 'audio';
    attachment.localPath = this.wavPath;
    attachment.displayName = this.wavPath.substring(this.wavPath.lastIndexOf('/') + 1);
    attachment.sizeBytes = fileIo.statSync(this.wavPath).size;
    attachment.createdAt = Date.now();
    this.wavPath = '';
    return attachment;
  }

  async cancel(): Promise<void> {
    this.recording = false;
    if (this.capturer) {
      const capturer = this.capturer;
      this.capturer = undefined;
      try {
        await capturer.stop();
      } catch (error) {
      }
      capturer.off('readData', this.readCallback);
      await capturer.release();
    }
    this.closePcm(true);
    if (this.wavPath && fileIo.accessSync(this.wavPath)) {
      fileIo.unlinkSync(this.wavPath);
    }
    this.wavPath = '';
  }

  private writeWav(): void {
    const output = fileIo.openSync(this.wavPath,
      fileIo.OpenMode.CREATE | fileIo.OpenMode.WRITE_ONLY | fileIo.OpenMode.TRUNC);
    const input = fileIo.openSync(this.pcmPath, fileIo.OpenMode.READ_ONLY);
    try {
      fileIo.writeSync(output.fd, this.createWavHeader(this.dataBytes));
      const buffer = new ArrayBuffer(64 * 1024);
      let readBytes = fileIo.readSync(input.fd, buffer);
      while (readBytes > 0) {
        fileIo.writeSync(output.fd, buffer, { length: readBytes });
        readBytes = fileIo.readSync(input.fd, buffer);
      }
      fileIo.fsyncSync(output.fd);
    } finally {
      fileIo.closeSync(input);
      fileIo.closeSync(output);
    }
  }

  private createWavHeader(dataBytes: number): ArrayBuffer {
    const header = new ArrayBuffer(44);
    const view = new DataView(header);
    this.writeAscii(view, 0, 'RIFF');
    view.setUint32(4, dataBytes + 36, true);
    this.writeAscii(view, 8, 'WAVE');
    this.writeAscii(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, CHANNELS, true);
    view.setUint32(24, SAMPLE_RATE, true);
    view.setUint32(28, SAMPLE_RATE * CHANNELS * BITS_PER_SAMPLE / 8, true);
    view.setUint16(32, CHANNELS * BITS_PER_SAMPLE / 8, true);
    view.setUint16(34, BITS_PER_SAMPLE, true);
    this.writeAscii(view, 36, 'data');
    view.setUint32(40, dataBytes, true);
    return header;
  }

  private writeAscii(view: DataView, offset: number, value: string): void {
    for (let index = 0; index < value.length; index++) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  }

  private closePcm(remove: boolean): void {
    if (this.pcmFile) {
      fileIo.closeSync(this.pcmFile);
      this.pcmFile = undefined;
    }
    if (remove && this.pcmPath && fileIo.accessSync(this.pcmPath)) {
      fileIo.unlinkSync(this.pcmPath);
      this.pcmPath = '';
    }
  }

  private safeName(value: string): string {
    return value.replace(/[^a-zA-Z0-9._-]/g, '-').substring(0, 96);
  }
}

import { common } from '@kit.AbilityKit';
import { camera, cameraPicker } from '@kit.CameraKit';
import { fileIo, fileUri, picker } from '@kit.CoreFileKit';
import { photoAccessHelper } from '@kit.MediaLibraryKit';
import { AttachmentRecord } from '../domain/AppTypes';

export class MediaService {
  private static instance: MediaService = new MediaService();

  static getInstance(): MediaService {
    return MediaService.instance;
  }

  async pickImages(context: common.UIAbilityContext, sessionId: string): Promise<AttachmentRecord[]> {
    const options = new photoAccessHelper.PhotoSelectOptions();
    options.MIMEType = photoAccessHelper.PhotoViewMIMETypes.IMAGE_TYPE;
    options.maxSelectNumber = 4;
    const result = await new photoAccessHelper.PhotoViewPicker().select(options);
    const attachments: AttachmentRecord[] = [];
    for (let index = 0; index < result.photoUris.length; index++) {
      attachments.push(this.copyToHistory(context, sessionId, result.photoUris[index], 'image', 'jpg', index));
    }
    return attachments;
  }

  async capturePhoto(context: common.UIAbilityContext, sessionId: string): Promise<AttachmentRecord[]> {
    const directory = `${context.filesDir}/history/${this.safeName(sessionId)}`;
    if (!fileIo.accessSync(directory)) {
      fileIo.mkdirSync(directory, true);
    }
    const fileName = `image-${Date.now()}-camera.jpg`;
    const targetPath = `${directory}/${fileName}`;
    const file = fileIo.openSync(targetPath, fileIo.OpenMode.CREATE | fileIo.OpenMode.WRITE_ONLY);
    fileIo.closeSync(file);
    const profile = new cameraPicker.PickerProfile();
    profile.cameraPosition = camera.CameraPosition.CAMERA_POSITION_BACK;
    profile.saveUri = fileUri.getUriFromPath(targetPath);
    const result = await cameraPicker.pick(context, [cameraPicker.PickerMediaType.PHOTO], profile);
    if (result.resultCode !== 0 || !fileIo.accessSync(targetPath) || fileIo.statSync(targetPath).size === 0) {
      if (fileIo.accessSync(targetPath)) {
        fileIo.unlinkSync(targetPath);
      }
      return [];
    }
    return [this.createAttachment(targetPath, 'image', fileName)];
  }

  async pickVideo(context: common.UIAbilityContext, sessionId: string): Promise<AttachmentRecord[]> {
    const options = new photoAccessHelper.PhotoSelectOptions();
    options.MIMEType = photoAccessHelper.PhotoViewMIMETypes.VIDEO_TYPE;
    options.maxSelectNumber = 1;
    const result = await new photoAccessHelper.PhotoViewPicker().select(options);
    if (result.photoUris.length === 0) {
      return [];
    }
    return [this.copyToHistory(context, sessionId, result.photoUris[0], 'video', 'mp4', 0)];
  }

  async pickAudio(context: common.UIAbilityContext, sessionId: string): Promise<AttachmentRecord[]> {
    const options = new picker.AudioSelectOptions();
    options.maxSelectNumber = 1;
    const uris = await new picker.AudioViewPicker(context).select(options);
    if (uris.length === 0) {
      return [];
    }
    const attachment = this.copyToHistory(context, sessionId, uris[0], 'audio', 'wav', 0);
    if (!this.isWavFile(attachment.localPath)) {
      this.deleteAttachment(context, attachment);
      throw new Error('当前 Omni 音频输入仅支持 PCM WAV 文件');
    }
    return [attachment];
  }

  deleteAttachment(context: common.UIAbilityContext, attachment: AttachmentRecord): void {
    const historyRoot = `${context.filesDir}/history/`;
    if (attachment.localPath.startsWith(historyRoot) && fileIo.accessSync(attachment.localPath)) {
      fileIo.unlinkSync(attachment.localPath);
    }
  }

  buildPrompt(attachments: AttachmentRecord[], text: string): string {
    let prompt = '';
    attachments.forEach((attachment: AttachmentRecord) => {
      if (attachment.mediaType === 'image') {
        prompt += `<img>${attachment.localPath}</img>`;
      } else if (attachment.mediaType === 'audio') {
        prompt += `<audio>${attachment.localPath}</audio>`;
      } else if (attachment.mediaType === 'video') {
        prompt += `<video>${attachment.localPath}</video>`;
      }
    });
    return prompt + text;
  }

  fileSize(path: string): number {
    try {
      return fileIo.statSync(path).size;
    } catch (error) {
      return 0;
    }
  }

  ensureHistoryDirectory(context: common.UIAbilityContext, sessionId: string): string {
    const directory = `${context.filesDir}/history/${this.safeName(sessionId)}`;
    if (!fileIo.accessSync(directory)) {
      fileIo.mkdirSync(directory, true);
    }
    return directory;
  }

  private copyToHistory(context: common.UIAbilityContext, sessionId: string, uri: string, mediaType: string,
    extension: string, index: number): AttachmentRecord {
    const directory = `${context.filesDir}/history/${this.safeName(sessionId)}`;
    if (!fileIo.accessSync(directory)) {
      fileIo.mkdirSync(directory, true);
    }
    const fileName = `${mediaType}-${Date.now()}-${index}.${extension}`;
    const targetPath = `${directory}/${fileName}`;
    const source = fileIo.openSync(uri, fileIo.OpenMode.READ_ONLY);
    try {
      fileIo.copyFileSync(source.fd, targetPath);
    } finally {
      fileIo.closeSync(source);
    }
    return this.createAttachment(targetPath, mediaType, fileName);
  }

  private isWavFile(path: string): boolean {
    const file = fileIo.openSync(path, fileIo.OpenMode.READ_ONLY);
    const header = new ArrayBuffer(12);
    try {
      if (fileIo.readSync(file.fd, header) !== 12) {
        return false;
      }
    } finally {
      fileIo.closeSync(file);
    }
    const bytes = new Uint8Array(header);
    return bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46 &&
      bytes[8] === 0x57 && bytes[9] === 0x41 && bytes[10] === 0x56 && bytes[11] === 0x45;
  }

  private createAttachment(path: string, mediaType: string, fileName: string): AttachmentRecord {
    const attachment = new AttachmentRecord();
    attachment.mediaType = mediaType;
    attachment.localPath = path;
    attachment.displayName = fileName;
    attachment.sizeBytes = fileIo.statSync(path).size;
    attachment.createdAt = Date.now();
    return attachment;
  }

  private safeName(value: string): string {
    return value.replace(/[^a-zA-Z0-9._-]/g, '-').substring(0, 96);
  }
}

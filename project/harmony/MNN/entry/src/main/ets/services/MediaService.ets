import { tr } from './I18n';
import { common } from '@kit.AbilityKit';
import { camera, cameraPicker } from '@kit.CameraKit';
import { fileIo, fileUri, picker } from '@kit.CoreFileKit';
import { photoAccessHelper } from '@kit.MediaLibraryKit';
import { image } from '@kit.ImageKit';
import { AttachmentRecord } from '../domain/AppTypes';

export class MediaService {
  private static instance: MediaService = new MediaService();
  // Full-resolution camera photos (12 MP+) make the vision prefill take minutes on CPU and push
  // RSS towards the per-app limit; the Android client downsamples attachments, so do we.
  private static readonly MAX_IMAGE_EDGE: number = 768;

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
      const record = this.copyToHistory(context, sessionId, result.photoUris[index], 'image', 'jpg', index);
      await this.downscaleImage(record.localPath);
      record.sizeBytes = this.fileSize(record.localPath);
      attachments.push(record);
    }
    return attachments;
  }

  async downscaleImage(path: string): Promise<void> {
    let source: image.ImageSource | undefined = undefined;
    try {
      source = image.createImageSource(path);
      const info = await source.getImageInfo();
      const maxEdge = Math.max(info.size.width, info.size.height);
      if (maxEdge <= MediaService.MAX_IMAGE_EDGE) {
        return;
      }
      const scale = MediaService.MAX_IMAGE_EDGE / maxEdge;
      const pixelMap = await source.createPixelMap({
        desiredSize: {
          width: Math.max(1, Math.round(info.size.width * scale)),
          height: Math.max(1, Math.round(info.size.height * scale))
        }
      });
      const packer = image.createImagePacker();
      const tmpPath = `${path}.tmp.jpg`;
      let file: fileIo.File | undefined = undefined;
      try {
        file = fileIo.openSync(tmpPath, fileIo.OpenMode.CREATE | fileIo.OpenMode.WRITE_ONLY);
        await packer.packToFile(pixelMap, file.fd, { format: 'image/jpeg', quality: 90 });
      } finally {
        if (file) {
          try {
            fileIo.closeSync(file);
          } catch (closeError) {
          }
        }
        try {
          packer.release();
        } catch (releaseError) {
        }
        try {
          pixelMap.release();
        } catch (releaseError) {
        }
      }
      fileIo.renameSync(tmpPath, path);
    } catch (error) {
      // Keep the original attachment when downsampling fails for any reason.
    } finally {
      if (source) {
        try {
          source.release();
        } catch (error) {
        }
      }
    }
  }

  async capturePhoto(context: common.UIAbilityContext, sessionId: string): Promise<AttachmentRecord[]> {
    const directory = `${context.filesDir}/history/${this.safeName(sessionId)}`;
    if (!fileIo.accessSync(directory)) {
      fileIo.mkdirSync(directory, true);
    }
    const fileName = `image-${Date.now()}-camera.jpg`;
    const targetPath = `${directory}/${fileName}`;
    // PickerProfile instances expose read-only properties at runtime; pass a plain literal.
    // The saveUri target must exist beforehand, otherwise the picker returns success with
    // an empty resultUri and writes nothing.
    const pre = fileIo.openSync(targetPath, fileIo.OpenMode.CREATE | fileIo.OpenMode.WRITE_ONLY);
    fileIo.closeSync(pre);
    const profile: cameraPicker.PickerProfile = {
      cameraPosition: camera.CameraPosition.CAMERA_POSITION_BACK,
      saveUri: fileUri.getUriFromPath(targetPath)
    };
    const result = await cameraPicker.pick(context, [cameraPicker.PickerMediaType.PHOTO], profile);
    console.info(`[Media] cameraPicker result=${result.resultCode} uri=${result.resultUri || 'none'}`);
    if (result.resultCode !== 0 || !fileIo.accessSync(targetPath) || fileIo.statSync(targetPath).size === 0) {
      if (fileIo.accessSync(targetPath)) {
        fileIo.unlinkSync(targetPath);
      }
      return [];
    }
    await this.downscaleImage(targetPath);
    return [this.createAttachment(targetPath, 'image', fileName)];
  }

  async pickVideo(context: common.UIAbilityContext, sessionId: string): Promise<AttachmentRecord[]> {
    const options = new picker.DocumentSelectOptions();
    options.maxSelectNumber = 1;
    options.fileSuffixFilters = ['.mp4', '.mov'];
    const uris = await new picker.DocumentViewPicker(context).select(options);
    if (uris.length === 0) {
      return [];
    }
    return [this.copyToHistory(context, sessionId, uris[0], 'video', 'mp4', 0)];
  }

  async pickAudio(context: common.UIAbilityContext, sessionId: string): Promise<AttachmentRecord[]> {
    // DocumentViewPicker browses Downloads reliably; the audio picker depends on media indexing
// and shows an empty page on some devices. The engine only accepts PCM WAV anyway.
    const options = new picker.DocumentSelectOptions();
    options.maxSelectNumber = 1;
    options.fileSuffixFilters = ['.wav'];
    const uris = await new picker.DocumentViewPicker(context).select(options);
    if (uris.length === 0) {
      return [];
    }
    const attachment = this.copyToHistory(context, sessionId, uris[0], 'audio', 'wav', 0);
    if (!this.isWavFile(attachment.localPath)) {
      this.deleteAttachment(context, attachment);
      throw new Error(tr('当前 Omni 音频输入仅支持 PCM WAV 文件'));
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

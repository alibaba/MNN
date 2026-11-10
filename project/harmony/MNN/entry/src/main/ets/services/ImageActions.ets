import { common } from '@kit.AbilityKit';
import { fileIo, fileUri } from '@kit.CoreFileKit';
import { photoAccessHelper } from '@kit.MediaLibraryKit';
import { systemShare } from '@kit.ShareKit';
import { uniformTypeDescriptor } from '@kit.ArkData';

/**
 * Save and share actions for chat images, mirroring the Android client's image menu.
 * Saving goes through the system "save to gallery" confirmation, so no media permission is needed.
 */
export class ImageActions {
  static async saveToGallery(context: common.UIAbilityContext, path: string): Promise<void> {
    const extension = ImageActions.extensionOf(path);
    const uri = fileUri.getUriFromPath(path);
    const config: photoAccessHelper.PhotoCreationConfig = {
      title: `MNN_Chat_${Date.now()}`,
      fileNameExtension: extension,
      photoType: photoAccessHelper.PhotoType.IMAGE
    };
    const helper = photoAccessHelper.getPhotoAccessHelper(context);
    const targets = await helper.showAssetsCreationDialog([uri], [config]);
    if (targets.length === 0) {
      throw new Error('未选择保存位置');
    }
    ImageActions.copyFile(path, targets[0]);
  }

  static async share(context: common.UIAbilityContext, path: string): Promise<void> {
    const extension = ImageActions.extensionOf(path);
    const record: systemShare.SharedRecord = {
      utd: extension === 'png' ? uniformTypeDescriptor.UniformDataType.PNG :
        (extension === 'jpg' ? uniformTypeDescriptor.UniformDataType.JPEG :
          uniformTypeDescriptor.UniformDataType.IMAGE),
      uri: fileUri.getUriFromPath(path),
      title: path.substring(path.lastIndexOf('/') + 1)
    };
    const controller = new systemShare.ShareController(new systemShare.SharedData(record));
    await controller.show(context, {
      selectionMode: systemShare.SelectionMode.SINGLE,
      previewMode: systemShare.SharePreviewMode.DETAIL
    });
  }

  private static extensionOf(path: string): string {
    const dot = path.lastIndexOf('.');
    const extension = dot >= 0 ? path.substring(dot + 1).toLowerCase() : '';
    if (extension === 'jpg' || extension === 'jpeg') {
      return 'jpg';
    }
    return extension === 'png' ? 'png' : 'jpg';
  }

  private static copyFile(sourcePath: string, targetUri: string): void {
    const source = fileIo.openSync(sourcePath, fileIo.OpenMode.READ_ONLY);
    let target: fileIo.File | undefined = undefined;
    try {
      target = fileIo.openSync(targetUri, fileIo.OpenMode.READ_WRITE | fileIo.OpenMode.TRUNC);
      const buffer = new ArrayBuffer(64 * 1024);
      while (true) {
        const read = fileIo.readSync(source.fd, buffer);
        if (read <= 0) {
          break;
        }
        fileIo.writeSync(target.fd, buffer, { length: read });
        if (read < buffer.byteLength) {
          break;
        }
      }
    } finally {
      fileIo.closeSync(source);
      if (target !== undefined) {
        fileIo.closeSync(target);
      }
    }
  }
}
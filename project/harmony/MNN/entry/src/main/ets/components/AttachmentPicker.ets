import { common } from '@kit.AbilityKit';
import { AttachmentRecord, ModelCapabilities } from '../domain/AppTypes';
import { MediaService } from '../services/MediaService';
import { VoiceRecorderService } from '../services/VoiceRecorderService';

@Component
export struct AttachmentPicker {
  @Link attachments: AttachmentRecord[];
  @Link error: string;
  @Prop sessionId: string;
  @Prop capabilities: ModelCapabilities;
  @State private recording: boolean = false;

  private pickImages(): void {
    if (!this.capabilities.vision) {
      return;
    }
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    MediaService.getInstance().pickImages(context, this.sessionId).then((items: AttachmentRecord[]) => {
      this.attachments = this.attachments.concat(items);
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private capturePhoto(): void {
    if (!this.capabilities.vision) {
      return;
    }
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    MediaService.getInstance().capturePhoto(context, this.sessionId).then((items: AttachmentRecord[]) => {
      this.attachments = this.attachments.concat(items);
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private pickAudio(): void {
    if (!this.capabilities.audioInput) {
      return;
    }
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    MediaService.getInstance().pickAudio(context, this.sessionId).then((items: AttachmentRecord[]) => {
      this.attachments = this.attachments.filter((item: AttachmentRecord) => item.mediaType !== 'audio').concat(items);
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private toggleRecording(): void {
    if (!this.capabilities.audioInput || !this.sessionId) {
      return;
    }
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    if (!this.recording) {
      VoiceRecorderService.getInstance().start(context, this.sessionId).then(() => {
        this.recording = true;
      }).catch((error: Error) => {
        this.error = error.message;
      });
      return;
    }
    VoiceRecorderService.getInstance().stop().then((attachment: AttachmentRecord) => {
      this.recording = false;
      this.attachments = this.attachments.filter((item: AttachmentRecord) => item.mediaType !== 'audio')
        .concat([attachment]);
    }).catch((error: Error) => {
      this.recording = false;
      this.error = error.message;
    });
  }

  private pickVideo(): void {
    if (!this.capabilities.videoInput) {
      return;
    }
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    MediaService.getInstance().pickVideo(context, this.sessionId).then((items: AttachmentRecord[]) => {
      this.attachments = this.attachments.filter((item: AttachmentRecord) => item.mediaType !== 'video').concat(items);
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private remove(index: number): void {
    const attachment = this.attachments[index];
    if (attachment) {
      const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
      MediaService.getInstance().deleteAttachment(context, attachment);
    }
    this.attachments = this.attachments.filter((_item: AttachmentRecord, itemIndex: number) => itemIndex !== index);
  }

  build() {
    Column({ space: 8 }) {
      Row({ space: 8 }) {
        Button('图片')
          .height(30)
          .fontSize(12)
          .enabled(this.capabilities.vision && !!this.sessionId)
          .onClick(() => this.pickImages())
        Button('拍照')
          .height(30)
          .fontSize(12)
          .enabled(this.capabilities.vision && !!this.sessionId)
          .onClick(() => this.capturePhoto())
        Button('音频')
          .height(30)
          .fontSize(12)
          .enabled(this.capabilities.audioInput && !!this.sessionId)
          .onClick(() => this.pickAudio())
        Button(this.recording ? '停止录音' : '录音')
          .height(30)
          .fontSize(12)
          .fontColor(this.recording ? '#C62828' : '#2864DC')
          .enabled(this.capabilities.audioInput && !!this.sessionId)
          .onClick(() => this.toggleRecording())
        Button('视频')
          .height(30)
          .fontSize(12)
          .enabled(this.capabilities.videoInput && !!this.sessionId)
          .onClick(() => this.pickVideo())
        if (!this.capabilities.vision && !this.capabilities.audioInput) {
          Text(this.capabilities.attachmentReason)
            .fontSize(11)
            .fontColor('#8A9099')
            .layoutWeight(1)
        } else if (!this.capabilities.videoInput) {
          Text(this.capabilities.videoReason)
            .fontSize(11)
            .fontColor('#8A9099')
            .layoutWeight(1)
        }
      }
      .width('100%')

      if (this.attachments.length > 0) {
        Scroll() {
          Row({ space: 8 }) {
            ForEach(this.attachments, (attachment: AttachmentRecord, index: number) => {
              Stack({ alignContent: Alignment.TopEnd }) {
                if (attachment.mediaType === 'image') {
                  Image(`file://${attachment.localPath}`)
                    .width(72)
                    .height(72)
                    .objectFit(ImageFit.Cover)
                    .borderRadius(8)
                } else {
                  Text(attachment.mediaType === 'audio' ? '音频' : '视频')
                    .fontSize(12)
                    .textAlign(TextAlign.Center)
                    .width(72)
                    .height(72)
                    .backgroundColor('#EEF3FF')
                    .borderRadius(8)
                }
                Button('×')
                  .width(24)
                  .height(24)
                  .fontSize(12)
                  .backgroundColor('#CC20242A')
                  .fontColor(Color.White)
                  .onClick(() => this.remove(index))
              }
            }, (attachment: AttachmentRecord) => attachment.localPath)
          }
        }
        .scrollable(ScrollDirection.Horizontal)
        .scrollBar(BarState.Off)
        .width('100%')
      }
    }
    .width('100%')
  }
}

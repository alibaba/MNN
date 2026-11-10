import { ModelCapabilities } from '../domain/AppTypes';

/**
 * Bottom sheet shown from the chat input "+" button, mirroring the Android attachment panel.
 * Only the entry points supported by the loaded model's capability flags are rendered.
 */
@CustomDialog
export struct AttachmentSheet {
  controller?: CustomDialogController;
  capabilities: ModelCapabilities = new ModelCapabilities();
  sessionReady: boolean = false;
  recording: boolean = false;
  onPickImages: () => void = () => {};
  onCapturePhoto: () => void = () => {};
  onPickAudio: () => void = () => {};
  onToggleRecording: () => void = () => {};
  onPickVideo: () => void = () => {};

  private choose(action: () => void): void {
    this.controller?.close();
    action();
  }

  @Builder
  private option(label: string, enabled: boolean, action: () => void) {
    Button(label)
      .layoutWeight(1)
      .height(44)
      .fontSize(14)
      .fontColor(enabled ? '#2864DC' : '#9AA0A8')
      .backgroundColor(enabled ? '#EEF3FF' : '#F1F3F5')
      .enabled(enabled)
      .onClick(() => this.choose(action))
  }

  build() {
    Column({ space: 12 }) {
      Text('添加附件')
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .fontColor('#171A1F')
        .width('100%')

      if (this.capabilities.vision) {
        Row({ space: 10 }) {
          this.option('相册', this.sessionReady, () => this.onPickImages())
          this.option('拍照', this.sessionReady, () => this.onCapturePhoto())
        }
        .width('100%')
      }

      if (this.capabilities.audioInput) {
        Row({ space: 10 }) {
          this.option('音频文件', this.sessionReady, () => this.onPickAudio())
          this.option(this.recording ? '停止录音' : '录音', this.sessionReady,
            () => this.onToggleRecording())
        }
        .width('100%')
      }

      if (this.capabilities.videoInput) {
        this.option('视频', this.sessionReady, () => this.onPickVideo())
      }

      if (!this.capabilities.vision && !this.capabilities.audioInput && !this.capabilities.videoInput) {
        Text(this.capabilities.attachmentReason || '当前模型不支持多媒体输入')
          .fontSize(13)
          .fontColor('#6B7280')
          .width('100%')
      }

      Button('取消')
        .width('100%')
        .height(40)
        .fontSize(15)
        .fontColor('#5F6670')
        .backgroundColor('#F1F3F5')
        .onClick(() => this.controller?.close())
    }
    .width('100%')
    .padding(20)
    .backgroundColor(Color.White)
    .borderRadius({ topLeft: 16, topRight: 16 })
  }
}
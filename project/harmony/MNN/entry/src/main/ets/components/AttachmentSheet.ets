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
      .fontColor(enabled ? $r('app.color.brand') : $r('app.color.text_disabled'))
      .backgroundColor(enabled ? $r('app.color.brand_tint') : $r('app.color.surface_muted'))
      .enabled(enabled)
      .onClick(() => this.choose(action))
  }

  build() {
    Column({ space: 12 }) {
      Text('添加附件')
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
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
        Row({ space: 10 }) {
          this.option('视频', this.sessionReady, () => this.onPickVideo())
        }
        .width('100%')
      }

      if (!this.capabilities.vision && !this.capabilities.audioInput && !this.capabilities.videoInput) {
        Text(this.capabilities.attachmentReason || '当前模型不支持多媒体输入')
          .fontSize(13)
          .fontColor($r('app.color.text_secondary'))
          .width('100%')
      }

      Button('取消')
        .width('100%')
        .height(40)
        .fontSize(15)
        .fontColor($r('app.color.text_muted'))
        .backgroundColor($r('app.color.surface_muted'))
        .onClick(() => this.controller?.close())
    }
    .width('100%')
    .padding(20)
    .backgroundColor($r('app.color.surface'))
    .borderRadius({ topLeft: 16, topRight: 16 })
  }
}
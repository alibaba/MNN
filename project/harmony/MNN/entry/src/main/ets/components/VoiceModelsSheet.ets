import { tr } from '../services/I18n';
import { common } from '@kit.AbilityKit';
import { VoiceInputCallbacks, VoiceInputService } from '../services/VoiceInputService';
import { TtsService } from '../services/TtsService';

/**
 * Voice model management, mirroring Android's voice-model market sheet. On HarmonyOS the
 * ASR/TTS engines ship with HMS Speech Kit (no separate model downloads), so this sheet shows
 * engine status and offers live tests instead of a download list.
 */
@CustomDialog
export struct VoiceModelsSheet {
  controller?: CustomDialogController;
  @State private asrText: string = '';
  @State private asrBusy: boolean = false;
  @State private ttsBusy: boolean = false;

  private async testAsr(): Promise<void> {
    if (this.asrBusy) {
      await VoiceInputService.getInstance().stop();
      this.asrBusy = false;
      return;
    }
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    const callbacks = new VoiceInputCallbacks();
    this.asrText = '';
    this.asrBusy = true;
    callbacks.onPartial = (text: string) => {
      this.asrText = text;
    };
    callbacks.onFinal = (text: string) => {
      this.asrText = text;
      this.asrBusy = false;
    };
    callbacks.onState = (listening: boolean, message: string) => {
      if (!listening && !message) {
        this.asrBusy = false;
      }
    };
    try {
      await VoiceInputService.getInstance().start(context, callbacks);
    } catch (error) {
      this.asrBusy = false;
      this.asrText = (error as Error).message;
    }
  }

  private async testTts(): Promise<void> {
    if (this.ttsBusy) {
      TtsService.getInstance().stop();
      return;
    }
    this.ttsBusy = true;
    try {
      await TtsService.getInstance().speak(tr('你好，这是系统语音合成测试。'), () => {
        this.ttsBusy = false;
      });
    } catch (error) {
      this.ttsBusy = false;
      this.asrText = (error as Error).message;
    }
  }

  build() {
    Column({ space: 14 }) {
      Text(tr('语音模型管理'))
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .fontColor($r('app.color.text_primary'))
        .width('100%')

      Text(tr('鸿蒙端语音能力由系统 Speech Kit 提供（离线中文识别 + 系统 TTS），无需单独下载语音模型；Omni 模型自带 talker 语音回复。'))
        .fontSize(12)
        .lineHeight(19)
        .fontColor($r('app.color.text_secondary'))
        .width('100%')

      Row() {
        Column({ space: 3 }) {
          Text(tr('离线语音识别 (zh-CN)'))
            .fontSize(14)
            .fontColor($r('app.color.text_body'))
          Text(tr('HMS Speech Kit · 流式'))
            .fontSize(11)
            .fontColor($r('app.color.text_tertiary'))
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        Button(this.asrBusy ? tr('停止') : tr('测试'))
          .height(32)
          .fontSize(12)
          .onClick(() => this.testAsr())
      }
      .width('100%')

      if (this.asrText) {
        Text(this.asrText)
          .fontSize(13)
          .fontColor($r('app.color.text_body_alt'))
          .width('100%')
          .padding(10)
          .borderRadius(8)
          .backgroundColor($r('app.color.surface_alt'))
      }

      Row() {
        Column({ space: 3 }) {
          Text(tr('系统语音合成 (TTS)'))
            .fontSize(14)
            .fontColor($r('app.color.text_body'))
          Text(tr('HMS Speech Kit · zh-CN'))
            .fontSize(11)
            .fontColor($r('app.color.text_tertiary'))
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        Button(this.ttsBusy ? tr('停止') : tr('测试'))
          .height(32)
          .fontSize(12)
          .onClick(() => this.testTts())
      }
      .width('100%')

      Button(tr('关闭'))
        .width('100%')
        .height(40)
        .fontSize(14)
        .backgroundColor($r('app.color.surface_muted'))
        .fontColor($r('app.color.text_body'))
        .onClick(() => {
          if (this.asrBusy) {
            VoiceInputService.getInstance().stop().catch(() => {
            });
          }
          TtsService.getInstance().stop();
          this.controller?.close();
        })
    }
    .padding(20)
    .backgroundColor($r('app.color.surface'))
    .borderRadius({ topLeft: 20, topRight: 20 })
  }
}

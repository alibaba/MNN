import { tr } from '../../services/I18n';
import { common } from '@kit.AbilityKit';
import { fileIo, picker } from '@kit.CoreFileKit';
import { VoiceInputCallbacks, VoiceInputService } from '../../services/VoiceInputService';
import { TtsService } from '../../services/TtsService';
import { VideoFrameExtractor } from '../../services/VideoFrameExtractor';

/**
 * Developer test page mirroring Android's debug activities (ASR / TTS / video decoder tests).
 */
@Component
export struct DebugPage {
  onExit: () => void = () => {};
  @State private asrText: string = '';
  @State private asrOn: boolean = false;
  @State private ttsText: string = '你好，这是调试合成测试。';
  @State private ttsBusy: boolean = false;
  @State private videoInfo: string = '';
  @State private framePath: string = '';
  @State private busy: boolean = false;
  @State private error: string = '';

  private async toggleAsr(): Promise<void> {
    if (this.asrOn) {
      this.asrOn = false;
      await VoiceInputService.getInstance().stop();
      return;
    }
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    const callbacks = new VoiceInputCallbacks();
    this.asrText = '';
    callbacks.onPartial = (text: string) => {
      this.asrText = text;
    };
    callbacks.onFinal = (text: string) => {
      this.asrText = text;
      this.asrOn = false;
    };
    callbacks.onState = (listening: boolean, message: string) => {
      if (!listening && !message) {
        this.asrOn = false;
      }
      if (message && !listening) {
        this.error = message;
      }
    };
    try {
      await VoiceInputService.getInstance().start(context, callbacks);
      this.asrOn = true;
    } catch (error) {
      this.error = (error as Error).message;
    }
  }

  private async speakTts(): Promise<void> {
    if (this.ttsBusy) {
      TtsService.getInstance().stop();
      return;
    }
    this.ttsBusy = true;
    try {
      await TtsService.getInstance().speak(this.ttsText || tr('你好，这是调试合成测试。'), () => {
        this.ttsBusy = false;
      });
    } catch (error) {
      this.ttsBusy = false;
      this.error = (error as Error).message;
    }
  }

  private async pickVideoAndExtract(): Promise<void> {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    const options = new picker.DocumentSelectOptions();
    options.maxSelectNumber = 1;
    try {
      this.busy = true;
      const uris = await new picker.DocumentViewPicker(context).select(options);
      if (uris.length === 0) {
        this.busy = false;
        return;
      }
      const cachePath = `${context.cacheDir}/debug-video-${Date.now()}.mp4`;
      const src = fileIo.openSync(uris[0], fileIo.OpenMode.READ_ONLY);
      const dst = fileIo.openSync(cachePath, fileIo.OpenMode.CREATE | fileIo.OpenMode.WRITE_ONLY);
      fileIo.copyFileSync(src.fd, dst.fd);
      fileIo.closeSync(src);
      fileIo.closeSync(dst);
      const frames = await VideoFrameExtractor.getInstance().extract(cachePath, context.cacheDir, 4, 320);
      this.videoInfo = `${frames.length}${tr('帧已抽取')}`;
      this.framePath = frames.length > 0 ? frames[0] : '';
    } catch (error) {
      this.error = (error as Error).message;
    } finally {
      this.busy = false;
    }
  }

  @Builder
  private card(title: string) {
    Text(title)
      .fontSize(15)
      .fontWeight(FontWeight.Medium)
      .fontColor($r('app.color.text_primary'))
      .width('100%')
  }

  build() {
    Column() {
      Row({ space: 10 }) {
        Button('←')
          .width(40)
          .height(40)
          .fontSize(18)
          .padding(0)
          .fontColor($r('app.color.text_primary'))
          .backgroundColor($r('app.color.surface_muted'))
          .onClick(() => this.onExit())
        Text(tr('调试测试'))
          .fontSize(24)
          .fontWeight(FontWeight.Bold)
          .layoutWeight(1)
      }
      .width('100%')
      .padding({ left: 16, right: 20, top: 14, bottom: 12 })

      Scroll() {
        Column({ space: 14 }) {
          Column({ space: 10 }) {
            this.card(tr('语音识别测试'))
            Text(this.asrOn ? tr('正在听…') : (this.asrText || tr('点击开始，对着手机说话')))
              .fontSize(13)
              .fontColor($r('app.color.text_secondary'))
              .width('100%')
            Button(this.asrOn ? tr('停止') : tr('开始'))
              .height(34)
              .fontSize(13)
              .onClick(() => this.toggleAsr())
          }
          .width('100%')
          .padding(14)
          .borderRadius(12)
          .backgroundColor($r('app.color.surface'))
          .alignItems(HorizontalAlign.Start)

          Column({ space: 10 }) {
            this.card(tr('语音合成测试'))
            TextInput({ text: this.ttsText })
              .onChange((value: string) => {
                this.ttsText = value;
              })
            Button(this.ttsBusy ? tr('停止') : tr('合成'))
              .height(34)
              .fontSize(13)
              .onClick(() => this.speakTts())
          }
          .width('100%')
          .padding(14)
          .borderRadius(12)
          .backgroundColor($r('app.color.surface'))
          .alignItems(HorizontalAlign.Start)

          Column({ space: 10 }) {
            this.card(tr('视频抽帧测试'))
            if (this.framePath) {
              Image(`file://${this.framePath}`)
                .width(120)
                .height(120)
                .objectFit(ImageFit.Cover)
                .borderRadius(8)
            }
            Text(this.videoInfo || tr('选择视频并抽取 4 帧'))
              .fontSize(13)
              .fontColor($r('app.color.text_secondary'))
              .width('100%')
            Button(tr('选择视频并抽取'))
              .height(34)
              .fontSize(13)
              .enabled(!this.busy)
              .onClick(() => this.pickVideoAndExtract())
          }
          .width('100%')
          .padding(14)
          .borderRadius(12)
          .backgroundColor($r('app.color.surface'))
          .alignItems(HorizontalAlign.Start)

          if (this.error) {
            Text(`${tr('错误：')}${this.error}`)
              .fontSize(13)
              .fontColor($r('app.color.danger'))
              .width('100%')
          }
        }
        .padding({ left: 16, right: 16, bottom: 24 })
      }
      .layoutWeight(1)
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background'))
  }
}

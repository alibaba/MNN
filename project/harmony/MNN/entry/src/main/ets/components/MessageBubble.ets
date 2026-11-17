import { tr } from '../services/I18n';
import { promptAction } from '@kit.ArkUI';
import pasteboard from '@ohos.pasteboard';
import { AttachmentRecord, GenerationMetrics, GenerationState, MessageRole } from '../domain/AppTypes';
import { AudioPlaybackService } from '../services/AudioPlaybackService';
import { ImagePreviewDialog } from './ImagePreviewDialog';
import { MetricsPanel } from './MetricsPanel';
import { ReasoningPanel } from './ReasoningPanel';
import { RichMessage } from './RichMessage';

@Observed
export class ChatMessageView {
  id: number;
  recordId: string;
  role: MessageRole;
  rawContent: string;
  content: string;
  reasoning: string;
  isThinking: boolean;
  state: GenerationState;
  metrics: GenerationMetrics;
  attachments: AttachmentRecord[];
  timeLabel: string = '';

  constructor(id: number, recordId: string, role: MessageRole, content: string,
    state: GenerationState = GenerationState.COMPLETE) {
    this.id = id;
    this.recordId = recordId;
    this.role = role;
    this.rawContent = content;
    this.content = content;
    this.reasoning = '';
    this.isThinking = false;
    this.state = state;
    this.metrics = new GenerationMetrics();
    this.attachments = [];
    this.timeLabel = ChatMessageView.formatTime(Date.now());
  }

  static formatTime(epochMs: number): string {
    const date = new Date(epochMs);
    const pad = (value: number): string => value < 10 ? `0${value}` : `${value}`;
    return `${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }
}

@Component
export struct MessageBubble {
  @ObjectLink message: ChatMessageView;
  @Prop showMetrics: boolean = true;
  @Prop showTimestamp: boolean = false;
  @StorageLink('audioPlaying') private audioPlaying: boolean = false;
  onReportIssue: () => void = () => {};
  onPlayAudio: (attachment: AttachmentRecord) => void = () => {};
  onToggleMetrics: () => void = () => {};
  onShare: (text: string) => void = () => {};

  private openImage(path: string): void {
    let controller: CustomDialogController | null = null;
    controller = new CustomDialogController({
      builder: ImagePreviewDialog({ path: path }),
      autoCancel: true,
      alignment: DialogAlignment.Center,
      customStyle: true,
      width: '100%',
      height: '100%',
      cornerRadius: 0,
      backgroundColor: Color.Black
    });
    controller.open();
  }

  private copyText(text: string, label: string): void {
    if (!text) {
      return;
    }
    try {
      const data = pasteboard.createData(pasteboard.MIMETYPE_TEXT_PLAIN, text);
      pasteboard.getSystemPasteboard().setData(data).then(() => {
        promptAction.showToast({ message: `${label}${tr('已复制')}` });
      }).catch(() => {
        promptAction.showToast({ message: tr('复制失败') });
      });
    } catch (error) {
      promptAction.showToast({ message: tr('复制失败') });
    }
  }

  private audioInfo(): string {
    const lines: string[] = [];
    this.message.attachments.forEach((attachment: AttachmentRecord) => {
      if (attachment.mediaType === 'audio') {
        lines.push(`${attachment.displayName} · ${(attachment.sizeBytes / 1024).toFixed(1)} KB`);
      }
    });
    if (lines.length === 0) {
      lines.push(this.message.rawContent);
    }
    return lines.join('\n');
  }

  @Builder
  private messageMenu() {
    Menu() {
      MenuItem({ content: tr('复制文本') })
        .onClick(() => this.copyText(this.message.content || this.message.rawContent, tr('文本')))
      if (this.message.role === MessageRole.USER) {
        if (this.message.attachments.some((attachment: AttachmentRecord) =>
          attachment.mediaType === 'audio')) {
          MenuItem({ content: tr('复制音频信息') })
            .onClick(() => this.copyText(this.audioInfo(), tr('音频信息')))
        }
      } else {
        MenuItem({ content: tr('上报问题') })
          .onClick(() => this.onReportIssue())
      }
    }
  }

  @Builder
  private attachmentRow() {
    Flex({ wrap: FlexWrap.Wrap, justifyContent: this.message.role === MessageRole.USER ?
      FlexAlign.End : FlexAlign.Start }) {
      ForEach(this.message.attachments, (attachment: AttachmentRecord) => {
        if (attachment.mediaType === 'image') {
          Image(`file://${attachment.localPath}`)
            .width(92)
            .height(92)
            .objectFit(ImageFit.Cover)
            .borderRadius(8)
            .margin({ right: 6, bottom: 6 })
            .onClick(() => this.openImage(attachment.localPath))
        } else {
          Row({ space: 8 }) {
            Text(attachment.mediaType === 'audio' ? `${tr('音频 · ')}${attachment.displayName}` :
              `${tr('视频 · ')}${attachment.displayName}`)
              .fontSize(12)
              .fontColor($r('app.color.brand'))
            if (attachment.mediaType === 'audio') {
              Button(this.audioPlaying ? tr('停止') : tr('播放'))
                .height(26)
                .fontSize(11)
                .padding({ left: 10, right: 10 })
                .backgroundColor(this.audioPlaying ? $r('app.color.danger') : $r('app.color.brand'))
                .fontColor(Color.White)
                .onClick(() => {
                  if (this.audioPlaying) {
                    AudioPlaybackService.getInstance().stop().catch(() => {
                    });
                  } else {
                    this.onPlayAudio(attachment);
                  }
                })
            }
          }
          .padding(8)
          .borderRadius(8)
          .backgroundColor($r('app.color.brand_tint'))
          .margin({ right: 6, bottom: 6 })
        }
      }, (attachment: AttachmentRecord) => attachment.id || attachment.localPath)
    }
    .width('100%')
  }

  build() {
    // Mirrors the Android chat list: centered timestamps, plain assistant text on the
    // background and a filled pill for user messages (no avatars, no cards).
    Column({ space: 10 }) {
      if (this.showTimestamp && this.message.timeLabel) {
        Text(this.message.timeLabel)
          .fontSize(12)
          .fontColor($r('app.color.text_disabled'))
          .width('100%')
          .textAlign(TextAlign.Center)
      }
      if (this.message.role === MessageRole.USER) {
        Column({ space: 8 }) {
          if (this.message.attachments.length > 0) {
            this.attachmentRow()
          }
          if (this.message.content) {
            Text(this.message.content)
              .fontSize(15)
              .lineHeight(22)
              .fontColor(Color.White)
              .padding({ left: 14, right: 14, top: 10, bottom: 10 })
              .borderRadius(16)
              .backgroundColor($r('app.color.bubble_user'))
          }
        }
        .width('100%')
        .alignItems(HorizontalAlign.End)
      } else {
        Column({ space: 8 }) {
          if (this.message.attachments.length > 0) {
            this.attachmentRow()
          }
          if (this.message.reasoning) {
            ReasoningPanel({
              reasoning: this.message.reasoning,
              isThinking: this.message.isThinking
            })
          }
          if (this.message.content) {
            RichMessage({ content: this.message.content })
          } else if (this.message.state === GenerationState.GENERATING) {
            Row({ space: 8 }) {
              LoadingProgress()
                .width(18)
                .height(18)
              Text(this.message.isThinking ? tr('正在思考') : tr('正在生成'))
                .fontSize(15)
                .fontColor($r('app.color.text_secondary'))
            }
            .width('100%')
          } else {
            Text(tr('未生成内容'))
              .fontSize(15)
              .fontColor($r('app.color.text_secondary'))
          }
          if (this.showMetrics) {
            MetricsPanel({ metrics: this.message.metrics, state: this.message.state })
          }
          if (this.message.content && this.message.state !== GenerationState.GENERATING) {
            // Android-style per-message action row.
            Row({ space: 16 }) {
              Text(tr('上报'))
                .fontSize(12)
                .fontColor($r('app.color.text_tertiary'))
                .onClick(() => this.onReportIssue())
              Text(tr('指标'))
                .fontSize(12)
                .fontColor($r('app.color.text_tertiary'))
                .onClick(() => this.onToggleMetrics())
              if (this.message.attachments.some((attachment: AttachmentRecord) =>
                attachment.mediaType === 'audio')) {
                Text(this.audioPlaying ? tr('停止') : tr('重放'))
                  .fontSize(12)
                  .fontColor($r('app.color.text_tertiary'))
                  .onClick(() => {
                    if (this.audioPlaying) {
                      AudioPlaybackService.getInstance().stop().catch(() => {
                      });
                      return;
                    }
                    const audio = this.message.attachments.find((attachment: AttachmentRecord) =>
                      attachment.mediaType === 'audio');
                    if (audio) {
                      this.onPlayAudio(audio);
                    }
                  })
              }
              Text(tr('分享'))
                .fontSize(12)
                .fontColor($r('app.color.text_tertiary'))
                .onClick(() => this.onShare(this.message.content))
            }
            .width('100%')
          }
        }
        .width('100%')
        .alignItems(HorizontalAlign.Start)
      }
    }
    .width('100%')
    .bindContextMenu(this.messageMenu, ResponseType.LongPress)
  }
}

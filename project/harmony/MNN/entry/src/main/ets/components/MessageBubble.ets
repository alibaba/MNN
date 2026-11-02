import { AttachmentRecord, GenerationMetrics, GenerationState, MessageRole } from '../domain/AppTypes';
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
  }
}

@Component
export struct MessageBubble {
  @ObjectLink message: ChatMessageView;

  build() {
    Row() {
      if (this.message.role === MessageRole.ASSISTANT) {
        Text('M')
          .fontSize(14)
          .fontWeight(FontWeight.Bold)
          .fontColor(Color.White)
          .textAlign(TextAlign.Center)
          .width(34)
          .height(34)
          .borderRadius(17)
          .backgroundColor('#2864DC')
          .margin({ right: 10 })
      }

      Column() {
        Text(this.message.role === MessageRole.ASSISTANT ? 'MNN' : '我')
          .fontSize(13)
          .fontColor('#7A7F87')
          .width('100%')
          .margin({ bottom: 5 })
        if (this.message.attachments.length > 0) {
          Flex({ wrap: FlexWrap.Wrap }) {
            ForEach(this.message.attachments, (attachment: AttachmentRecord) => {
              if (attachment.mediaType === 'image') {
                Image(`file://${attachment.localPath}`)
                  .width(92)
                  .height(92)
                  .objectFit(ImageFit.Cover)
                  .borderRadius(8)
                  .margin({ right: 6, bottom: 6 })
              } else {
                Text(attachment.mediaType === 'audio' ? `音频 · ${attachment.displayName}` :
                  `视频 · ${attachment.displayName}`)
                  .fontSize(12)
                  .fontColor('#2864DC')
                  .backgroundColor('#EEF3FF')
                  .padding(8)
                  .borderRadius(8)
                  .margin({ right: 6, bottom: 6 })
              }
            }, (attachment: AttachmentRecord) => attachment.id || attachment.localPath)
          }
          .width('100%')
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
            Text(this.message.isThinking ? '正在思考' : '正在生成')
              .fontSize(15)
              .fontColor('#6B7280')
          }
          .width('100%')
        } else {
          Text('未生成内容')
            .fontSize(15)
            .fontColor('#6B7280')
        }
        if (this.message.role === MessageRole.ASSISTANT) {
          MetricsPanel({ metrics: this.message.metrics, state: this.message.state })
        }
      }
      .padding(14)
      .borderRadius(14)
      .backgroundColor(this.message.role === MessageRole.ASSISTANT ? '#F3F5F8' : '#E8F0FF')
      .layoutWeight(1)

      if (this.message.role === MessageRole.USER) {
        Text('我')
          .fontSize(14)
          .fontWeight(FontWeight.Bold)
          .fontColor(Color.White)
          .textAlign(TextAlign.Center)
          .width(34)
          .height(34)
          .borderRadius(17)
          .backgroundColor('#20242A')
          .margin({ left: 10 })
      }
    }
    .width('100%')
    .alignItems(VerticalAlign.Top)
    .padding({
      left: this.message.role === MessageRole.USER ? 44 : 0,
      right: this.message.role === MessageRole.ASSISTANT ? 44 : 0
    })
  }
}

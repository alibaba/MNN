import { common } from '@kit.AbilityKit';
import { ChatRepository } from '../../data/ChatRepository';
import { ChatSessionRecord } from '../../domain/AppTypes';
import { AppInitializer } from '../../services/AppInitializer';

@Component
export struct HistoryPage {
  @Link requestedSessionId: string;
  @Link selectedTab: number;
  @State private sessions: ChatSessionRecord[] = [];
  @State private error: string = '';
  @State private renameId: string = '';
  @State private renameText: string = '';

  aboutToAppear(): void {
    this.refresh();
  }

  private refresh(): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    AppInitializer.initialize(context).then(() => {
      return ChatRepository.getInstance().listSessions();
    }).then((sessions: ChatSessionRecord[]) => {
      this.sessions = sessions;
      this.error = '';
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private openSession(session: ChatSessionRecord): void {
    this.requestedSessionId = session.id;
    this.selectedTab = 0;
  }

  private beginRename(session: ChatSessionRecord): void {
    this.renameId = session.id;
    this.renameText = session.title;
  }

  private saveRename(): void {
    const title = this.renameText.trim();
    if (!this.renameId || !title) {
      return;
    }
    ChatRepository.getInstance().renameSession(this.renameId, title).then(() => {
      this.renameId = '';
      this.renameText = '';
      this.refresh();
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private deleteSession(sessionId: string): void {
    if (this.requestedSessionId === sessionId) {
      this.error = '当前会话不能删除，请先打开其他会话或新建对话';
      return;
    }
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    ChatRepository.getInstance().deleteSession(sessionId, context.filesDir).then(() => {
      if (this.requestedSessionId === sessionId) {
        this.requestedSessionId = '';
      }
      this.refresh();
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  build() {
    Column() {
      Row() {
        Text('历史对话')
          .fontSize(24)
          .fontWeight(FontWeight.Bold)
          .layoutWeight(1)
        Button('刷新')
          .height(36)
          .fontSize(13)
          .onClick(() => this.refresh())
      }
      .width('100%')
      .padding(18)

      if (this.renameId) {
        Row({ space: 8 }) {
          TextInput({ text: this.renameText, placeholder: '输入会话名称' })
            .layoutWeight(1)
            .onChange((value: string) => {
              this.renameText = value;
            })
          Button('保存')
            .onClick(() => this.saveRename())
          Button('取消')
            .backgroundColor($r('app.color.divider_alt'))
            .fontColor($r('app.color.text_body'))
            .onClick(() => {
              this.renameId = '';
            })
        }
        .padding({ left: 16, right: 16, bottom: 12 })
      }

      if (this.sessions.length === 0) {
        Column({ space: 10 }) {
          Text('暂无历史对话')
            .fontSize(18)
            .fontColor($r('app.color.text_body'))
          Text('在聊天页发送第一条消息后，会话会自动保存在本机。')
            .fontSize(14)
            .fontColor($r('app.color.text_tertiary_alt'))
            .textAlign(TextAlign.Center)
        }
        .layoutWeight(1)
        .justifyContent(FlexAlign.Center)
        .padding(32)
      } else {
        List({ space: 10 }) {
          ForEach(this.sessions, (session: ChatSessionRecord) => {
            ListItem() {
              Row({ space: 10 }) {
                Column({ space: 5 }) {
                  Text(session.title)
                    .fontSize(16)
                    .fontWeight(FontWeight.Medium)
                    .fontColor($r('app.color.text_primary'))
                    .maxLines(1)
                  Text(`${new Date(session.updatedAt).toLocaleString()} · ${session.modelId}`)
                    .fontSize(12)
                    .fontColor($r('app.color.text_tertiary'))
                }
                .alignItems(HorizontalAlign.Start)
                .layoutWeight(1)
                .onClick(() => this.openSession(session))

                Button('重命名')
                  .height(32)
                  .fontSize(12)
                  .backgroundColor($r('app.color.brand_tint'))
                  .fontColor($r('app.color.brand'))
                  .onClick(() => this.beginRename(session))
                Button('删除')
                  .height(32)
                  .fontSize(12)
                  .enabled(this.requestedSessionId !== session.id)
                  .backgroundColor($r('app.color.danger_tint'))
                  .fontColor($r('app.color.danger'))
                  .onClick(() => this.deleteSession(session.id))
              }
              .padding(14)
              .borderRadius(12)
              .backgroundColor($r('app.color.surface'))
            }
          }, (session: ChatSessionRecord) => session.id)
        }
        .layoutWeight(1)
        .padding({ left: 16, right: 16, bottom: 16 })
      }

      if (this.error) {
        Text(`错误：${this.error}`)
          .fontSize(13)
          .fontColor($r('app.color.danger'))
          .padding(16)
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background'))
  }
}

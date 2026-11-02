import { common } from '@kit.AbilityKit';
import { ApiServer, ApiServerStatus } from '../../services/ApiServer';

@Component
export struct ApiConsolePage {
  @Watch('onConfigurationChanged') @Link activeModelId: string;
  @Watch('onConfigurationChanged') @Link settingsRevision: number;
  @State private status: ApiServerStatus = new ApiServerStatus();
  @State private portText: string = '8080';
  @State private error: string = '';
  @State private busy: boolean = false;

  aboutToAppear(): void {
    this.status = ApiServer.getInstance().getStatus();
  }

  private onConfigurationChanged(): void {
    if (this.status.running) {
      this.stop();
      this.error = '模型已切换，请重新启动 API 服务';
    }
  }

  private start(): void {
    const port = Number(this.portText);
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.busy = true;
    ApiServer.getInstance().start(context, this.activeModelId, port).then((status: ApiServerStatus) => {
      this.status = status;
      this.error = '';
    }).catch((error: Error) => {
      this.error = error.message;
    }).finally(() => {
      this.busy = false;
    });
  }

  private stop(): void {
    this.busy = true;
    ApiServer.getInstance().stop().then(() => {
      this.status = ApiServer.getInstance().getStatus();
    }).catch((error: Error) => {
      this.error = error.message;
    }).finally(() => {
      this.busy = false;
    });
  }

  private refresh(): void {
    this.status = ApiServer.getInstance().getStatus();
  }

  build() {
    Column() {
      Text('本地 API')
        .fontSize(24)
        .fontWeight(FontWeight.Bold)
        .width('100%')
        .padding({ left: 20, right: 20, top: 18, bottom: 12 })

      Column({ space: 12 }) {
        Row() {
          Text('状态')
            .fontSize(15)
            .layoutWeight(1)
          Text(this.status.running ? '运行中' : '已停止')
            .fontSize(14)
            .fontColor(this.status.running ? '#168653' : '#6B7280')
        }
        Row() {
          Text('监听地址')
            .fontSize(15)
            .layoutWeight(1)
          Text(this.status.running ? `${this.status.address}:${this.status.port}` : '127.0.0.1')
            .fontSize(14)
            .fontColor('#4B5563')
        }
        TextInput({ text: this.portText, placeholder: '端口' })
          .type(InputType.Number)
          .enabled(!this.status.running && !this.busy)
          .onChange((value: string) => this.portText = value)
        Text('Bearer Token')
          .fontSize(13)
          .fontColor('#6B7280')
          .width('100%')
        Text(this.status.token || '启动后生成')
          .fontSize(12)
          .fontColor('#20242A')
          .copyOption(CopyOptions.LocalDevice)
          .width('100%')
          .padding(10)
          .backgroundColor('#F3F5F8')
          .borderRadius(8)
        Text(`模型：${this.activeModelId} · 连接：${this.status.connections} · 队列：${this.status.queueSize}`)
          .fontSize(12)
          .fontColor('#6B7280')
          .width('100%')
        Row({ space: 10 }) {
          Button(this.status.running ? '停止服务' : '启动服务')
            .layoutWeight(1)
            .enabled(!this.busy)
            .fontColor(this.status.running ? '#C62828' : Color.White)
            .backgroundColor(this.status.running ? '#FDECEC' : '#2864DC')
            .onClick(() => this.status.running ? this.stop() : this.start())
          Button('刷新状态')
            .layoutWeight(1)
            .enabled(!this.busy)
            .onClick(() => this.refresh())
        }
      }
      .width('100%')
      .padding(16)
      .backgroundColor(Color.White)
      .borderRadius(14)
      .margin({ left: 16, right: 16, bottom: 14 })

      Column({ space: 8 }) {
        Text('兼容端点')
          .fontSize(17)
          .fontWeight(FontWeight.Bold)
          .width('100%')
        Text('GET /v1/models')
        Text('GET /v1/queue')
        Text('POST /v1/chat/completions')
        Text('POST /v1/messages')
        Text('POST /v1/cancel')
        Text('仅监听 127.0.0.1，所有请求默认要求 Bearer Token；HDC 转发后仍需鉴权。')
          .fontSize(12)
          .lineHeight(19)
          .fontColor('#6B7280')
      }
      .width('100%')
      .padding(16)
      .backgroundColor(Color.White)
      .borderRadius(14)
      .margin({ left: 16, right: 16 })

      if (this.error) {
        Text(`错误：${this.error}`)
          .fontSize(13)
          .fontColor('#C62828')
          .padding(16)
      }
      Blank()
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#FAFBFC')
  }
}

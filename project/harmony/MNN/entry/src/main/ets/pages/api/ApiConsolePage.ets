import { common } from '@kit.AbilityKit';
import {
  API_AUTH_ENABLED_KEY,
  API_BIND_ALL_KEY,
  API_CORS_ENABLED_KEY,
  API_TOKEN_SETTING_KEY,
  ApiRequestLog,
  ApiServer,
  ApiServerStatus
} from '../../services/ApiServer';
import { SettingsStore } from '../../services/SettingsStore';

@Component
export struct ApiConsolePage {
  @Watch('onConfigurationChanged') @Link activeModelId: string;
  @Watch('onConfigurationChanged') @Link settingsRevision: number;
  @State private status: ApiServerStatus = new ApiServerStatus();
  @State private portText: string = '8080';
  @State private error: string = '';
  @State private busy: boolean = false;
  @State private bindAll: boolean = false;
  @State private authEnabled: boolean = true;
  @State private corsEnabled: boolean = false;
  @State private tokenText: string = '';

  aboutToAppear(): void {
    this.status = ApiServer.getInstance().getStatus();
    this.loadApiSettings();
  }

  private loadApiSettings(): void {
    const store = SettingsStore.getInstance();
    store.getBoolean(API_BIND_ALL_KEY, false).then((value: boolean) => this.bindAll = value);
    store.getBoolean(API_AUTH_ENABLED_KEY, true).then((value: boolean) => this.authEnabled = value);
    store.getBoolean(API_CORS_ENABLED_KEY, false).then((value: boolean) => this.corsEnabled = value);
    store.getString(API_TOKEN_SETTING_KEY, this.status.token).then((value: string) => this.tokenText = value);
  }

  private saveApiSettings(): void {
    const store = SettingsStore.getInstance();
    this.busy = true;
    Promise.all([
      store.putBoolean(API_BIND_ALL_KEY, this.bindAll),
      store.putBoolean(API_AUTH_ENABLED_KEY, this.authEnabled),
      store.putBoolean(API_CORS_ENABLED_KEY, this.corsEnabled),
      store.putString(API_TOKEN_SETTING_KEY, this.tokenText.trim())
    ]).then(() => {
      if (this.status.running) {
        return ApiServer.getInstance().stop();
      }
      return Promise.resolve();
    }).then(() => {
      this.status = ApiServer.getInstance().getStatus();
      if (this.status.running) {
        return Promise.resolve();
      }
      return Promise.resolve();
    }).then(() => {
      this.error = '设置已保存，重新启动服务后生效';
    }).catch((error: Error) => {
      this.error = error.message;
    }).finally(() => {
      this.busy = false;
    });
  }

  private resetApiSettings(): void {
    this.bindAll = false;
    this.authEnabled = true;
    this.corsEnabled = false;
    this.tokenText = '';
    SettingsStore.getInstance().getString(API_TOKEN_SETTING_KEY, '').then(() => {
      const store = SettingsStore.getInstance();
      return Promise.all([
        store.putBoolean(API_BIND_ALL_KEY, false),
        store.putBoolean(API_AUTH_ENABLED_KEY, true),
        store.putBoolean(API_CORS_ENABLED_KEY, false),
        store.putString(API_TOKEN_SETTING_KEY, '')
      ]);
    }).then(() => {
      this.error = '已恢复默认设置，重新启动服务后生效';
    }).catch((error: Error) => {
      this.error = error.message;
    });
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
    Scroll() {
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
        Row() {
          Text('监听范围')
            .fontSize(15)
            .layoutWeight(1)
          Text(this.bindAll ? '局域网' : '仅本机')
            .fontSize(14)
            .fontColor('#4B5563')
        }
        Row({ space: 10 }) {
          Button('仅本机')
            .layoutWeight(1)
            .height(34)
            .fontSize(13)
            .backgroundColor(this.bindAll ? '#F1F3F5' : '#EEF3FF')
            .fontColor(this.bindAll ? '#5F6670' : '#2864DC')
            .enabled(!this.status.running && !this.busy)
            .onClick(() => this.bindAll = false)
          Button('局域网')
            .layoutWeight(1)
            .height(34)
            .fontSize(13)
            .backgroundColor(this.bindAll ? '#EEF3FF' : '#F1F3F5')
            .fontColor(this.bindAll ? '#2864DC' : '#5F6670')
            .enabled(!this.status.running && !this.busy)
            .onClick(() => this.bindAll = true)
        }
        Row() {
          Text('要求 Bearer 鉴权')
            .fontSize(15)
            .layoutWeight(1)
          Toggle({ type: ToggleType.Switch, isOn: this.authEnabled })
            .enabled(!this.busy)
            .onChange((value: boolean) => this.authEnabled = value)
        }
        Text('API Key')
          .fontSize(13)
          .fontColor('#6B7280')
          .width('100%')
        TextInput({ text: this.tokenText, placeholder: '留空则由服务自动生成' })
          .enabled(!this.busy)
          .onChange((value: string) => this.tokenText = value)
        Row() {
          Text('允许跨域 (CORS)')
            .fontSize(15)
            .layoutWeight(1)
          Toggle({ type: ToggleType.Switch, isOn: this.corsEnabled })
            .enabled(!this.busy)
            .onChange((value: boolean) => this.corsEnabled = value)
        }
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
        Row({ space: 10 }) {
          Button('保存设置')
            .layoutWeight(1)
            .enabled(!this.busy)
            .backgroundColor('#EEF3FF')
            .fontColor('#2864DC')
            .onClick(() => this.saveApiSettings())
          Button('重置')
            .layoutWeight(1)
            .enabled(!this.busy)
            .backgroundColor('#F1F3F5')
            .fontColor('#5F6670')
            .onClick(() => this.resetApiSettings())
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
        Text(this.bindAll ? '监听 0.0.0.0，局域网内设备可访问；鉴权与 CORS 按上方设置生效。' :
          '仅监听 127.0.0.1，默认要求 Bearer Token；HDC 转发后仍需鉴权。')
          .fontSize(12)
          .lineHeight(19)
          .fontColor('#6B7280')
      }
      .width('100%')
      .padding(16)
      .backgroundColor(Color.White)
      .borderRadius(14)
      .margin({ left: 16, right: 16 })

      Column({ space: 8 }) {
        Row() {
          Text('请求日志')
            .fontSize(17)
            .fontWeight(FontWeight.Bold)
            .layoutWeight(1)
          Button('刷新')
            .height(30)
            .fontSize(12)
            .onClick(() => this.refresh())
        }
        .width('100%')
        if (this.status.requestLogs.length === 0) {
          Text('暂无请求，服务启动后在这里查看最近的 20 条记录')
            .fontSize(12)
            .fontColor('#6B7280')
            .width('100%')
        } else {
          ForEach(this.status.requestLogs, (log: ApiRequestLog) => {
            Text(`${log.time}  ${log.method} ${log.path}  → ${log.status}  ${log.durationMs}ms`)
              .fontSize(12)
              .fontColor(log.status >= 400 ? '#C62828' : '#303640')
              .fontFamily('monospace')
              .width('100%')
              .maxLines(1)
              .textOverflow({ overflow: TextOverflow.Ellipsis })
          }, (log: ApiRequestLog) => `${log.time}-${log.method}-${log.path}-${log.durationMs}`)
        }
      }
      .width('100%')
      .padding(16)
      .backgroundColor(Color.White)
      .borderRadius(14)
      .margin({ left: 16, right: 16, bottom: 14 })

      if (this.error) {
        Text(`错误：${this.error}`)
          .fontSize(13)
          .fontColor('#C62828')
          .padding(16)
      }
      }
      .width('100%')
      .padding({ bottom: 16 })
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#FAFBFC')
    .scrollBar(BarState.Auto)
  }
}

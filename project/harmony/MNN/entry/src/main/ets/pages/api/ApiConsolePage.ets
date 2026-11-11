import { tr } from '../../services/I18n';
import { common } from '@kit.AbilityKit';
import { promptAction } from '@kit.ArkUI';
import pasteboard from '@ohos.pasteboard';
import {
  API_AUTH_ENABLED_KEY,
  API_BIND_ALL_KEY,
  API_CORS_ENABLED_KEY,
  API_CORS_ORIGINS_KEY,
  API_HTTPS_URL_KEY,
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
  @State private corsOriginsText: string = '';
  @State private httpsUrl: boolean = false;
  @State private tokenText: string = '';

  aboutToAppear(): void {
    this.status = ApiServer.getInstance().getStatus();
    this.loadApiSettings();
  }

  private addressScheme(): string {
    return this.httpsUrl ? 'https' : 'http';
  }

  private copyAddress(): void {
    const address = `${this.addressScheme()}://${this.status.address}:${this.status.port}`;
    try {
      const data = pasteboard.createData(pasteboard.MIMETYPE_TEXT_PLAIN, address);
      pasteboard.getSystemPasteboard().setData(data).then(() => {
        promptAction.showToast({ message: tr('地址已复制') });
      }).catch(() => {
        promptAction.showToast({ message: tr('复制失败') });
      });
    } catch (error) {
      promptAction.showToast({ message: tr('复制失败') });
    }
  }

  private loadApiSettings(): void {
    const store = SettingsStore.getInstance();
    store.getBoolean(API_BIND_ALL_KEY, false).then((value: boolean) => this.bindAll = value);
    store.getBoolean(API_AUTH_ENABLED_KEY, true).then((value: boolean) => this.authEnabled = value);
    store.getBoolean(API_CORS_ENABLED_KEY, false).then((value: boolean) => this.corsEnabled = value);
    store.getString(API_CORS_ORIGINS_KEY, '').then((value: string) => this.corsOriginsText = value);
    store.getBoolean(API_HTTPS_URL_KEY, false).then((value: boolean) => this.httpsUrl = value);
    store.getString(API_TOKEN_SETTING_KEY, this.status.token).then((value: string) => this.tokenText = value);
  }

  private saveApiSettings(): void {
    const store = SettingsStore.getInstance();
    this.busy = true;
    Promise.all([
      store.putBoolean(API_BIND_ALL_KEY, this.bindAll),
      store.putBoolean(API_AUTH_ENABLED_KEY, this.authEnabled),
      store.putBoolean(API_CORS_ENABLED_KEY, this.corsEnabled),
      store.putString(API_CORS_ORIGINS_KEY, this.corsOriginsText.trim()),
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
      this.error = tr('设置已保存，重新启动服务后生效');
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
    this.corsOriginsText = '';
    this.httpsUrl = false;
    this.tokenText = '';
    SettingsStore.getInstance().getString(API_TOKEN_SETTING_KEY, '').then(() => {
      const store = SettingsStore.getInstance();
      return Promise.all([
        store.putBoolean(API_BIND_ALL_KEY, false),
        store.putBoolean(API_AUTH_ENABLED_KEY, true),
        store.putBoolean(API_CORS_ENABLED_KEY, false),
        store.putString(API_CORS_ORIGINS_KEY, ''),
        store.putBoolean(API_HTTPS_URL_KEY, false),
        store.putString(API_TOKEN_SETTING_KEY, '')
      ]);
    }).then(() => {
      this.error = tr('已恢复默认设置，重新启动服务后生效');
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private onConfigurationChanged(): void {
    if (this.status.running) {
      this.stop();
      this.error = tr('模型已切换，请重新启动 API 服务');
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
        Text(tr('本地 API'))
        .fontSize(24)
        .fontWeight(FontWeight.Bold)
        .width('100%')
        .padding({ left: 20, right: 20, top: 18, bottom: 12 })

      Column({ space: 12 }) {
        Row() {
          Text(tr('状态'))
            .fontSize(15)
            .layoutWeight(1)
          Text(this.status.running ? tr('运行中') : tr('已停止'))
            .fontSize(14)
            .fontColor(this.status.running ? $r('app.color.success') : $r('app.color.text_secondary'))
        }
        Row() {
          Text(tr('监听地址'))
            .fontSize(15)
            .layoutWeight(1)
          Text(this.status.running ?
            `${this.addressScheme()}://${this.status.address}:${this.status.port}` : '127.0.0.1')
            .fontSize(14)
            .fontColor($r('app.color.text_body'))
          if (this.status.running) {
            Button(tr('复制'))
              .fontSize(12)
              .fontColor($r('app.color.brand'))
              .backgroundColor(Color.Transparent)
              .height(28)
              .padding(0)
              .margin({ left: 10 })
              .onClick(() => this.copyAddress())
          }
        }
        TextInput({ text: this.portText, placeholder: tr('端口') })
          .type(InputType.Number)
          .enabled(!this.status.running && !this.busy)
          .onChange((value: string) => this.portText = value)
        Row() {
          Text(tr('监听范围'))
            .fontSize(15)
            .layoutWeight(1)
          Text(this.bindAll ? tr('局域网') : tr('仅本机'))
            .fontSize(14)
            .fontColor($r('app.color.text_body'))
        }
        Row({ space: 10 }) {
          Button(tr('仅本机'))
            .layoutWeight(1)
            .height(34)
            .fontSize(13)
            .backgroundColor(this.bindAll ? $r('app.color.surface_muted') : $r('app.color.brand_tint'))
            .fontColor(this.bindAll ? $r('app.color.text_muted') : $r('app.color.brand'))
            .enabled(!this.status.running && !this.busy)
            .onClick(() => this.bindAll = false)
          Button(tr('局域网'))
            .layoutWeight(1)
            .height(34)
            .fontSize(13)
            .backgroundColor(this.bindAll ? $r('app.color.brand_tint') : $r('app.color.surface_muted'))
            .fontColor(this.bindAll ? $r('app.color.brand') : $r('app.color.text_muted'))
            .enabled(!this.status.running && !this.busy)
            .onClick(() => this.bindAll = true)
        }
        Row() {
          Text(tr('要求 Bearer 鉴权'))
            .fontSize(15)
            .layoutWeight(1)
          Toggle({ type: ToggleType.Switch, isOn: this.authEnabled })
            .enabled(!this.busy)
            .onChange((value: boolean) => this.authEnabled = value)
        }
        Text('API Key')
          .fontSize(13)
          .fontColor($r('app.color.text_secondary'))
          .width('100%')
        TextInput({ text: this.tokenText, placeholder: tr('留空则由服务自动生成') })
          .enabled(!this.busy)
          .onChange((value: string) => this.tokenText = value)
        Row() {
          Text(tr('允许跨域 (CORS)'))
            .fontSize(15)
            .layoutWeight(1)
          Toggle({ type: ToggleType.Switch, isOn: this.corsEnabled })
            .enabled(!this.busy)
            .onChange((value: boolean) => this.corsEnabled = value)
        }
        if (this.corsEnabled) {
          TextInput({ text: this.corsOriginsText, placeholder: tr('允许的来源，逗号分隔；留空允许全部') })
            .enabled(!this.busy)
            .onChange((value: string) => this.corsOriginsText = value)
          Text(tr('例如 http://localhost:3000,https://example.com'))
            .fontSize(11)
            .fontColor($r('app.color.text_tertiary'))
            .width('100%')
        }
        Row() {
          Text(tr('客户端使用 HTTPS URL'))
            .fontSize(15)
            .layoutWeight(1)
          Toggle({ type: ToggleType.Switch, isOn: this.httpsUrl })
            .enabled(!this.busy)
            .onChange((value: boolean) => {
              this.httpsUrl = value;
              SettingsStore.getInstance().putBoolean(API_HTTPS_URL_KEY, value).catch((error: Error) => {
                this.error = error.message;
              });
            })
        }
        Text(tr('仅影响展示与复制的地址；服务本身仍是 HTTP，需要 TLS 时请在前面加反向代理。'))
          .fontSize(11)
          .fontColor($r('app.color.text_tertiary'))
          .width('100%')
        Text(`${tr('模型：')}${this.activeModelId}${tr(' · 连接：')}${this.status.connections}${tr(' · 队列：')}${this.status.queueSize}`)
          .fontSize(12)
          .fontColor($r('app.color.text_secondary'))
          .width('100%')
        if (this.status.running) {
          Text(this.status.backgroundActive ?
            tr('已申请后台运行，切到后台后仍可响应请求，系统会显示常驻通知。') :
            tr('后台常驻未启用，切到后台后服务可能被系统挂起。'))
            .fontSize(11)
            .fontColor(this.status.backgroundActive ? $r('app.color.success') : $r('app.color.warning'))
            .width('100%')
        }
        Row({ space: 10 }) {
          Button(this.status.running ? tr('停止服务') : tr('启动服务'))
            .layoutWeight(1)
            .enabled(!this.busy)
            .fontColor(this.status.running ? $r('app.color.danger') : Color.White)
            .backgroundColor(this.status.running ? $r('app.color.danger_tint') : $r('app.color.brand'))
            .onClick(() => this.status.running ? this.stop() : this.start())
          Button(tr('刷新状态'))
            .layoutWeight(1)
            .enabled(!this.busy)
            .onClick(() => this.refresh())
        }
        Row({ space: 10 }) {
          Button(tr('保存设置'))
            .layoutWeight(1)
            .enabled(!this.busy)
            .backgroundColor($r('app.color.brand_tint'))
            .fontColor($r('app.color.brand'))
            .onClick(() => this.saveApiSettings())
          Button(tr('重置'))
            .layoutWeight(1)
            .enabled(!this.busy)
            .backgroundColor($r('app.color.surface_muted'))
            .fontColor($r('app.color.text_muted'))
            .onClick(() => this.resetApiSettings())
        }
      }
      .width('100%')
      .padding(16)
      .backgroundColor($r('app.color.surface'))
      .borderRadius(14)
      .margin({ left: 16, right: 16, bottom: 14 })

      Column({ space: 8 }) {
        Text(tr('兼容端点'))
          .fontSize(17)
          .fontWeight(FontWeight.Bold)
          .width('100%')
        Text('GET /v1/models')
        Text('GET /v1/queue · GET /v1/queue/status · DELETE /v1/queue/clear')
        Text('POST /v1/chat/completions')
        Text('POST /v1/messages')
        Text('POST /v1/cancel')
        Text(this.bindAll ? tr('监听 0.0.0.0，局域网内设备可访问；鉴权与 CORS 按上方设置生效。') :
          tr('仅监听 127.0.0.1，默认要求 Bearer Token；HDC 转发后仍需鉴权。'))
          .fontSize(12)
          .lineHeight(19)
          .fontColor($r('app.color.text_secondary'))
      }
      .width('100%')
      .padding(16)
      .backgroundColor($r('app.color.surface'))
      .borderRadius(14)
      .margin({ left: 16, right: 16 })

      Column({ space: 8 }) {
        Row() {
          Text(tr('请求日志'))
            .fontSize(17)
            .fontWeight(FontWeight.Bold)
            .layoutWeight(1)
          Button(tr('刷新'))
            .height(30)
            .fontSize(12)
            .onClick(() => this.refresh())
        }
        .width('100%')
        if (this.status.requestLogs.length === 0) {
          Text(tr('暂无请求，服务启动后在这里查看最近的 20 条记录'))
            .fontSize(12)
            .fontColor($r('app.color.text_secondary'))
            .width('100%')
        } else {
          ForEach(this.status.requestLogs, (log: ApiRequestLog) => {
            Text(`${log.time}  ${log.method} ${log.path}  → ${log.status}  ${log.durationMs}ms`)
              .fontSize(12)
              .fontColor(log.status >= 400 ? $r('app.color.danger') : $r('app.color.text_body_alt'))
              .fontFamily('monospace')
              .width('100%')
              .maxLines(1)
              .textOverflow({ overflow: TextOverflow.Ellipsis })
          }, (log: ApiRequestLog) => `${log.time}-${log.method}-${log.path}-${log.durationMs}`)
        }
      }
      .width('100%')
      .padding(16)
      .backgroundColor($r('app.color.surface'))
      .borderRadius(14)
      .margin({ left: 16, right: 16, bottom: 14 })

      if (this.error) {
        Text(`${tr('错误：')}${this.error}`)
          .fontSize(13)
          .fontColor($r('app.color.danger'))
          .padding(16)
      }
      }
      .width('100%')
      .padding({ bottom: 16 })
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background'))
    .scrollBar(BarState.Auto)
  }
}

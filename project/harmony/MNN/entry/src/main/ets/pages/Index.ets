import { tr } from '../services/I18n';
import { common, Want } from '@kit.AbilityKit';
import { promptAction } from '@kit.ArkUI';
import bundleManager from '@ohos.bundle.bundleManager';
import deviceInfo from '@ohos.deviceInfo';
import pasteboard from '@ohos.pasteboard';
import { mnnVersion } from '../services/GenerationCoordinator';
import { ApiConsolePage } from './api/ApiConsolePage';
import { BenchmarkPage } from './benchmark/BenchmarkPage';
import { ChatPage } from './chat/ChatPage';
import { HistoryPage } from './history/HistoryPage';
import { MarketPage } from './market/MarketPage';
import { ModelsPage } from './models/ModelsPage';
import { SettingsPage } from './settings/SettingsPage';
import { PrivacyPolicySheet } from '../components/PrivacyPolicySheet';
import { ModelPickerSheet } from '../components/ModelPickerSheet';
import { ModelRepository } from '../data/ModelRepository';
import { ModelRecord, ModelState } from '../domain/AppTypes';
import { AppInitializer } from '../services/AppInitializer';
import { PRIVACY_POLICY_KEY, PRIVACY_POLICY_VERSION } from '../services/PrivacyPolicy';
import { ACTIVE_MODEL_ID_KEY, SettingsStore } from '../services/SettingsStore';

const VIEW_MAIN: string = 'main';
const VIEW_CHAT: string = 'chat';
const VIEW_SETTINGS: string = 'settings';
const VIEW_API: string = 'api';

/**
 * Mirrors the Android app's navigation: a main screen with a history drawer and three bottom
 * tabs (My Models / Market / Benchmark); chat, settings and the API console are full-screen
 * views pushed on top of the main screen instead of permanent tabs.
 */
@Entry
@Component
struct Index {
  @State private currentView: string = VIEW_MAIN;
  @State private apiReturnView: string = VIEW_MAIN;
  @State private selectedTab: number = 0;
  @State private drawerShown: boolean = false;
  @State private requestedSessionId: string = '';
  @State private activeModelId: string = 'legacy-model';
  @State private activeModelName: string = '';
  @State private settingsRevision: number = 0;

  aboutToAppear(): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    AppInitializer.initialize(context).then(async () => {
      this.activeModelId = await SettingsStore.getInstance().getString(ACTIVE_MODEL_ID_KEY, 'legacy-model');
      this.settingsRevision = await SettingsStore.getInstance().getSettingsRevision();
      this.refreshModelName();
      const agreedVersion = await SettingsStore.getInstance().getNumber(PRIVACY_POLICY_KEY, 0);
      if (agreedVersion < PRIVACY_POLICY_VERSION) {
        setTimeout(() => this.showPrivacyPolicy(), 150);
      }
    });
  }

  onBackPress(): boolean {
    if (this.currentView !== VIEW_MAIN) {
      this.showMain();
      return true;
    }
    if (this.drawerShown) {
      this.drawerShown = false;
      return true;
    }
    return false;
  }

  private showMain(): void {
    this.currentView = VIEW_MAIN;
    this.refreshModelName();
  }

  private refreshModelName(): void {
    ModelRepository.getInstance().listModels().then((models: ModelRecord[]) => {
      const active = models.find((model: ModelRecord) => model.id === this.activeModelId);
      this.activeModelName = active ? active.name : '';
    }).catch(() => {
    });
  }

  private showPrivacyPolicy(): void {
    let controller: CustomDialogController | null = null;
    controller = new CustomDialogController({
      builder: PrivacyPolicySheet({
        onAgree: () => {
          SettingsStore.getInstance().putNumber(PRIVACY_POLICY_KEY, PRIVACY_POLICY_VERSION).catch(() => {
          });
        },
        onDecline: () => {
          const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
          context.terminateSelf().catch(() => {
          });
        }
      }),
      autoCancel: false,
      alignment: DialogAlignment.Center,
      customStyle: true
    });
    controller.open();
  }

  private openModelPicker(): void {
    ModelRepository.getInstance().listModels().then((models: ModelRecord[]) => {
      const ready: ModelRecord[] = models.filter((model: ModelRecord) => model.state === ModelState.READY);
      if (ready.length === 0) {
        return;
      }
      let controller: CustomDialogController | null = null;
      controller = new CustomDialogController({
        builder: ModelPickerSheet({
          models: ready,
          activeModelId: this.activeModelId,
          onPick: (model: ModelRecord) => {
            SettingsStore.getInstance().putString(ACTIVE_MODEL_ID_KEY, model.id).then(() => {
              this.activeModelId = model.id;
              this.refreshModelName();
            }).catch(() => {
            });
          }
        }),
        autoCancel: true,
        alignment: DialogAlignment.Bottom,
        customStyle: true
      });
      controller.open();
    }).catch(() => {
    });
  }

  private openUrl(uri: string): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    const want: Want = {
      action: 'ohos.want.action.viewData',
      entities: ['entity.system.browsable'],
      uri: uri
    };
    context.startAbility(want).catch(() => {
    });
  }

  /** Mirrors the Android "report issue" flow: copy device/app diagnostics, open GitHub issues. */
  private reportIssue(): void {
    let appVersion = '';
    try {
      appVersion = bundleManager.getBundleInfoForSelfSync(bundleManager.BundleFlag.GET_BUNDLE_INFO_DEFAULT).versionName;
    } catch (error) {
      appVersion = '';
    }
    const info = [
      `app_version: ${appVersion}`,
      `mnn_version: ${mnnVersion()}`,
      `device: ${deviceInfo.brand} ${deviceInfo.productModel}`,
      `os: ${deviceInfo.osFullName}`,
      `model_id: ${this.activeModelId}`
    ].join('\n');
    try {
      const data = pasteboard.createData(pasteboard.MIMETYPE_TEXT_PLAIN, info);
      pasteboard.getSystemPasteboard().setData(data).then(() => {
        promptAction.showToast({ message: tr('调试信息已复制，正在打开问题页面') });
      }).catch(() => {
        promptAction.showToast({ message: tr('调试信息复制失败') });
      });
    } catch (error) {
      promptAction.showToast({ message: tr('调试信息复制失败') });
    }
    this.openUrl('https://github.com/alibaba/MNN/issues');
  }

  private openChatWithModel(modelId: string): void {
    this.activeModelId = modelId;
    this.requestedSessionId = '';
    this.currentView = VIEW_CHAT;
  }

  private openChatSession(sessionId: string): void {
    this.requestedSessionId = sessionId;
    this.drawerShown = false;
    this.currentView = VIEW_CHAT;
  }

  @Builder
  private mainToolbar() {
    // Android-style app bar: hamburger, centered title (model switcher), overflow menu.
    Row() {
      Button() {
        Text('☰')
          .fontSize(20)
          .fontColor($r('app.color.text_primary'))
      }
      .width(44)
      .height(44)
      .backgroundColor(Color.Transparent)
      .onClick(() => {
        this.drawerShown = !this.drawerShown;
      })

      Text(this.activeModelName || 'MNN Chat')
        .fontSize(20)
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.text_primary'))
        .textAlign(TextAlign.Center)
        .maxLines(1)
        .textOverflow({ overflow: TextOverflow.Ellipsis })
        .layoutWeight(1)
        .onClick(() => this.openModelPicker())

      Row({ space: 2 }) {
        Button() {
          Text('★')
            .fontSize(20)
            .fontColor($r('app.color.text_primary'))
        }
        .width(44)
        .height(44)
        .backgroundColor(Color.Transparent)
        .onClick(() => this.openUrl('https://github.com/alibaba/MNN'))

        Button() {
          Text('⋮')
            .fontSize(20)
            .fontColor($r('app.color.text_primary'))
        }
        .width(44)
        .height(44)
        .backgroundColor(Color.Transparent)
        .bindMenu([
        {
          value: tr('切换模型'),
          action: () => this.openModelPicker()
        },
        {
          value: tr('设置'),
          action: () => {
            this.currentView = VIEW_SETTINGS;
          }
        },
        {
          value: tr('给 MNN 点个 Star'),
          action: () => this.openUrl('https://github.com/alibaba/MNN')
        },
        {
          value: tr('上报问题'),
          action: () => this.reportIssue()
        }
        ])
      }
    }
    .width('100%')
    .padding({ left: 6, right: 6, top: 8, bottom: 8 })
    .backgroundColor($r('app.color.background'))
  }

  @Builder
  private mainView() {
    Stack({ alignContent: Alignment.Start }) {
      Column() {
        this.mainToolbar()
        Divider()
          .color($r('app.color.divider'))
        Tabs({ barPosition: BarPosition.End, index: this.selectedTab }) {
          TabContent() {
            ModelsPage({
              activeModelId: this.activeModelId,
              onOpenChat: (modelId: string) => this.openChatWithModel(modelId),
              onOpenSettings: () => {
                this.currentView = VIEW_SETTINGS;
              },
              onStar: () => this.openUrl('https://github.com/alibaba/MNN'),
              onReportIssue: () => this.reportIssue()
            })
          }
          .tabBar(tr('模型'))

          TabContent() {
            MarketPage()
          }
          .tabBar(tr('市场'))

          TabContent() {
            BenchmarkPage({
              activeModelId: this.activeModelId,
              settingsRevision: this.settingsRevision
            })
          }
          .tabBar(tr('测试'))
        }
        .barHeight(54)
        .onChange((index: number) => {
          this.selectedTab = index;
        })
        .layoutWeight(1)
        .width('100%')
      }
      .width('100%')
      .height('100%')

      // History drawer, mirroring the Android DrawerLayout: scrim + left panel.
      if (this.drawerShown) {
        Column()
          .width('100%')
          .height('100%')
          .backgroundColor($r('app.color.overlay_close'))
          .onClick(() => {
            this.drawerShown = false;
          })

        Column() {
          HistoryPage({
            requestedSessionId: this.requestedSessionId,
            visible: this.drawerShown,
            onOpenChatSession: (sessionId: string) => this.openChatSession(sessionId)
          })
        }
        .width(320)
        .height('100%')
        .backgroundColor($r('app.color.background'))
        .transition(TransitionEffect.move(TransitionEdge.START).animation({ duration: 200 }))
      }
    }
    .width('100%')
    .height('100%')
  }

  build() {
    Column() {
      if (this.currentView === VIEW_CHAT) {
        ChatPage({
          requestedSessionId: this.requestedSessionId,
          activeModelId: this.activeModelId,
          settingsRevision: this.settingsRevision,
          onExit: () => this.showMain(),
          onOpenBenchmark: () => {
            this.selectedTab = 2;
            this.showMain();
          },
          onOpenSettings: () => {
            this.currentView = VIEW_SETTINGS;
          },
          onOpenApi: () => {
            this.apiReturnView = VIEW_CHAT;
            this.currentView = VIEW_API;
          }
        })
      } else if (this.currentView === VIEW_SETTINGS) {
        SettingsPage({
          activeModelId: this.activeModelId,
          settingsRevision: this.settingsRevision,
          onExit: () => this.showMain(),
          onOpenApi: () => {
            this.apiReturnView = VIEW_SETTINGS;
            this.currentView = VIEW_API;
          }
        })
      } else if (this.currentView === VIEW_API) {
        ApiConsolePage({
          activeModelId: this.activeModelId,
          settingsRevision: this.settingsRevision,
          onExit: () => {
            this.currentView = this.apiReturnView;
          }
        })
      } else {
        this.mainView()
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor($r('app.color.background'))
  }
}

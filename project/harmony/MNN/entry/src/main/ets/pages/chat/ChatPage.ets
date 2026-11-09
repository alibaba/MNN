import { common } from '@kit.AbilityKit';
import pasteboard from '@ohos.pasteboard';
import { AttachmentRepository } from '../../data/AttachmentRepository';
import { ChatRepository } from '../../data/ChatRepository';
import { ModelRepository } from '../../data/ModelRepository';
import {
  AttachmentRecord,
  ChatMessageRecord,
  ChatSessionRecord,
  GenerationState,
  MessageRole,
  ModelCapabilities,
  ModelRecord,
  ModelRuntimeSettings,
  ModelState
} from '../../domain/AppTypes';
import { AppInitializer } from '../../services/AppInitializer';
import {
  GenerationCoordinator,
  NativeChatMessage,
  NativeGenerateRequest,
  NativeGenerateResult,
  NativeLoadResult,
  NativeTarget
} from '../../services/GenerationCoordinator';
import { MediaService } from '../../services/MediaService';
import { ReasoningParser } from '../../services/ReasoningParser';
import { ACTIVE_MODEL_ID_KEY, SettingsStore } from '../../services/SettingsStore';
import { AttachmentPicker } from '../../components/AttachmentPicker';
import { InfoSheet } from '../../components/InfoSheet';
import { ModelPickerSheet } from '../../components/ModelPickerSheet';
import { ChatMessageView, MessageBubble } from '../../components/MessageBubble';

const CURRENT_SESSION_KEY: string = 'current_session_id';
const LEGACY_MODEL_ID: string = 'legacy-model';

@Component
export struct ChatPage {
  @Watch('onRequestedSessionChanged') @Link requestedSessionId: string;
  @Watch('onActiveModelChanged') @Link activeModelId: string;
  @Watch('onSettingsRevisionChanged') @Link settingsRevision: number;
  @State private prompt: string = '';
  @State private messages: ChatMessageView[] = [];
  @State private error: string = '';
  @State private modelStatus: string = '正在初始化';
  @State private metrics: string = '';
  @State private isModelReady: boolean = false;
  @State private isGenerating: boolean = false;
  @State private currentSessionId: string = '';
  @State private followsLatest: boolean = true;
  @State private activeModelName: string = '';
  @State private pendingAttachments: AttachmentRecord[] = [];
  @State private modelCapabilities: ModelCapabilities = new ModelCapabilities();
  private messageScroller: Scroller = new Scroller();
  private nextMessageId: number = 1;
  private activeRequestId: string = '';
  private stopRequested: boolean = false;
  private streamingAnswer: string = '';
  private initialized: boolean = false;
  private loadMs: number = 0;
  private loadedModelId: string = '';
  private loadedSettingsRevision: number = -1;
  private pendingSessionId: string = '';
  private restoreSequence: number = 0;
  private modelSwitchPending: boolean = false;
  private settingsReloadPending: boolean = false;
  private modelLoading: boolean = false;
  private runtimeSettings: ModelRuntimeSettings = new ModelRuntimeSettings();
  private nativeTarget?: NativeTarget;

  aboutToAppear(): void {
    this.initialize();
  }

  private onRequestedSessionChanged(): void {
    if (!this.initialized || !this.requestedSessionId || this.requestedSessionId === this.currentSessionId) {
      return;
    }
    if (this.isGenerating) {
      this.pendingSessionId = this.requestedSessionId;
      this.stopGeneration();
      return;
    }
    this.restoreRequestedSession(this.requestedSessionId);
  }

  private restoreRequestedSession(sessionId: string): void {
    const restoreSequence = ++this.restoreSequence;
    this.restoreSession(sessionId, restoreSequence).then(() => this.loadModel()).catch((error: Error) => {
      if (restoreSequence === this.restoreSequence) {
        this.error = error.message;
      }
    });
  }

  private onActiveModelChanged(): void {
    if (!this.initialized || !this.activeModelId || this.activeModelId === this.loadedModelId) {
      return;
    }
    if (this.isGenerating) {
      this.modelSwitchPending = true;
      this.stopGeneration();
      return;
    }
    if (this.modelLoading) {
      this.modelSwitchPending = true;
      return;
    }
    this.applyActiveModel().catch((error: Error) => {
      this.error = error.message;
      this.modelStatus = '模型加载失败';
    });
  }

  private onSettingsRevisionChanged(): void {
    if (!this.initialized || this.settingsRevision === this.loadedSettingsRevision) {
      return;
    }
    if (this.isGenerating || this.modelLoading) {
      this.settingsReloadPending = true;
      return;
    }
    this.isModelReady = false;
    this.loadModel().catch((error: Error) => {
      this.error = error.message;
      this.modelStatus = '模型加载失败';
    });
  }

  private async applyActiveModel(): Promise<void> {
    this.isModelReady = false;
    await SettingsStore.getInstance().putString(ACTIVE_MODEL_ID_KEY, this.activeModelId);
    await this.loadModel();
    if (this.currentSessionId) {
      await ChatRepository.getInstance().updateSessionModel(this.currentSessionId, this.activeModelId);
    }
  }

  private initialize(): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    AppInitializer.initialize(context).then(() => {
      this.initialized = true;
      return this.restoreInitialSession();
    }).then(() => {
      return this.loadModel();
    }).catch((error: Error) => {
      this.error = error.message;
      this.modelStatus = '初始化失败';
    });
  }

  private async restoreInitialSession(): Promise<void> {
    const savedSessionId = await SettingsStore.getInstance().getString(CURRENT_SESSION_KEY);
    const targetSessionId = this.requestedSessionId || savedSessionId;
    const sessions = await ChatRepository.getInstance().listSessions();
    let canRestore = false;
    sessions.forEach((session: ChatSessionRecord) => {
      if (session.id === targetSessionId) {
        canRestore = true;
      }
    });
    if (canRestore) {
      await this.restoreSession(targetSessionId, ++this.restoreSequence);
    } else {
      await this.createNewSession();
    }
  }

  private async restoreSession(sessionId: string, restoreSequence: number): Promise<void> {
    const session = await ChatRepository.getInstance().getSession(sessionId);
    if (!session) {
      throw new Error('会话不存在');
    }
    const records = await ChatRepository.getInstance().getMessages(sessionId);
    const restored: ChatMessageView[] = [];
    for (let index = 0; index < records.length; index++) {
      const record = records[index];
      if (record.state === GenerationState.GENERATING) {
        record.state = GenerationState.CANCELED;
        if (!record.displayContent) {
          record.displayContent = '上次生成意外中断';
          record.rawContent = record.displayContent;
        }
        await ChatRepository.getInstance().updateMessage(record);
      }
      const message = new ChatMessageView(
        this.nextMessageId++, record.id, record.role, record.displayContent, record.state
      );
      message.rawContent = record.rawContent;
      message.reasoning = record.reasoningContent;
      message.metrics = record.metrics;
      message.attachments = await AttachmentRepository.getInstance().getForMessage(record.id);
      restored.push(message);
    }
    if (restoreSequence !== this.restoreSequence) {
      return;
    }
    this.currentSessionId = sessionId;
    this.requestedSessionId = sessionId;
    this.messages = restored;
    this.metrics = '';
    this.error = '';
    this.clearPendingAttachments();
    this.activeModelId = session.modelId || LEGACY_MODEL_ID;
    await SettingsStore.getInstance().putString(ACTIVE_MODEL_ID_KEY, this.activeModelId);
    await SettingsStore.getInstance().putString(CURRENT_SESSION_KEY, sessionId);
  }

  private clearPendingAttachments(): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    this.pendingAttachments.forEach((attachment: AttachmentRecord) => {
      MediaService.getInstance().deleteAttachment(context, attachment);
    });
    this.pendingAttachments = [];
  }

  private async createNewSession(): Promise<void> {
    // Invalidate any in-flight session restore so it cannot overwrite the new conversation.
    ++this.restoreSequence;
    this.pendingSessionId = '';
    const session = await ChatRepository.getInstance().createSession(this.activeModelId || LEGACY_MODEL_ID);
    const greeting = await ChatRepository.getInstance().addMessage(
      session.id, MessageRole.ASSISTANT, '你好，我是运行在本机设备上的 MNN 助手。'
    );
    this.currentSessionId = session.id;
    this.requestedSessionId = session.id;
    this.messages = [new ChatMessageView(
      this.nextMessageId++, greeting.id, greeting.role, greeting.displayContent, greeting.state
    )];
    this.metrics = '';
    this.error = '';
    this.clearPendingAttachments();
    await SettingsStore.getInstance().putString(CURRENT_SESSION_KEY, session.id);
  }

  private async loadModel(): Promise<void> {
    if (this.modelLoading) {
      if (this.activeModelId !== this.loadedModelId) {
        this.modelSwitchPending = true;
      } else {
        this.settingsReloadPending = true;
      }
      return;
    }
    const targetModelId = this.activeModelId || LEGACY_MODEL_ID;
    const targetSettingsRevision = this.settingsRevision;
    if (this.loadedModelId === targetModelId && this.loadedSettingsRevision === targetSettingsRevision &&
      this.isModelReady) {
      return;
    }

    this.modelLoading = true;
    this.error = '';
    this.isModelReady = false;
    this.modelCapabilities = new ModelCapabilities();
    this.modelStatus = '正在加载模型';
    try {
      const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
      const model = await ModelRepository.getInstance().getModel(targetModelId);
      if (!model || model.state !== ModelState.READY) {
        throw new Error(`模型不可用：${targetModelId}`);
      }
      const settings = await SettingsStore.getInstance().getModelRuntimeSettings(model.id);
      const safeModelId = model.id.replace(/[^a-zA-Z0-9._-]/g, '-');
      const tmpPath = `${context.cacheDir}/llm-${safeModelId}`;
      const runtimeConfigJson = SettingsStore.getInstance().createRuntimeConfigJson(settings, tmpPath);
      const target: NativeTarget = {
        modelId: model.id,
        configPath: model.configPath,
        runtimeConfigJson: runtimeConfigJson
      };
      const result = await GenerationCoordinator.getInstance().prepare(target) as NativeLoadResult;
      this.loadMs = result.loadMs;
      this.runtimeSettings = settings;
      this.nativeTarget = target;
      this.loadedModelId = model.id;
      this.loadedSettingsRevision = targetSettingsRevision;
      this.activeModelName = model.name;
      this.isModelReady = true;
      this.modelStatus = result.reused ? '模型已就绪' : `模型已就绪 · ${(result.loadMs / 1000).toFixed(2)}s`;
      this.modelCapabilities = await ModelRepository.getInstance().getCapabilities(model.id);
      await ModelRepository.getInstance().markUsed(model.id);
    } finally {
      this.modelLoading = false;
      if (!this.isGenerating && this.modelSwitchPending) {
        this.modelSwitchPending = false;
        this.applyActiveModel().catch((error: Error) => {
          this.error = error.message;
          this.modelStatus = '模型加载失败';
        });
      } else if (!this.isGenerating && this.settingsReloadPending) {
        this.settingsReloadPending = false;
        this.isModelReady = false;
        this.loadModel().catch((error: Error) => {
          this.error = error.message;
          this.modelStatus = '模型加载失败';
        });
      }
    }
  }

  private fillTestPrompt(targetLength: number): void {
    const prefix = '请完整阅读下面的测试材料，概括核心观点，并从性能、内存、功耗和用户体验四个方面给出优化建议。测试材料：';
    const paragraph = '移动端大语言模型推理包含预填充与逐词解码两个阶段。预填充阶段需要并行处理较长的输入序列，计算吞吐、算子融合和线程调度会影响首字延迟；解码阶段每次只生成少量 token，模型权重和 KV 缓存的频繁访问会带来较高的内存带宽压力。实际优化还需要兼顾设备温度、持续功耗、峰值内存、响应速度和输出质量，并通过稳定的基准测试比较不同方案。';
    let value = prefix;
    while (value.length < targetLength) {
      value += paragraph;
    }
    this.prompt = value.substring(0, targetLength);
  }

  private buildNativeHistory(records: ChatMessageRecord[]): NativeChatMessage[] {
    const history: NativeChatMessage[] = [{
      role: MessageRole.SYSTEM,
      content: this.runtimeSettings.systemPrompt
    }];
    let hasUserMessage = false;
    records.forEach((record: ChatMessageRecord) => {
      if (record.state === GenerationState.GENERATING || record.state === GenerationState.FAILED) {
        return;
      }
      if (record.role === MessageRole.SYSTEM) {
        const systemPrompt = record.rawContent || record.displayContent;
        if (systemPrompt) {
          history[0].content = systemPrompt;
        }
        return;
      }
      if (record.role === MessageRole.USER) {
        const userContent = record.rawContent || record.displayContent;
        if (userContent) {
          history.push({ role: MessageRole.USER, content: userContent });
          hasUserMessage = true;
        }
        return;
      }
      if (hasUserMessage) {
        const assistantContent = record.displayContent;
        if (assistantContent) {
          history.push({ role: MessageRole.ASSISTANT, content: assistantContent });
        }
      }
    });
    return history;
  }

  private async runInference(): Promise<void> {
    const input = this.prompt.trim();
    const attachments = this.pendingAttachments.slice();
    if ((!input && attachments.length === 0) || !this.isModelReady || this.isGenerating || !this.currentSessionId) {
      return;
    }
    const displayInput = input || '已发送附件';
    const nativeInput = MediaService.getInstance().buildPrompt(attachments, input);

    this.error = '';
    this.metrics = '';
    this.isGenerating = true;
    this.stopRequested = false;
    this.streamingAnswer = '';
    const requestId = GenerationCoordinator.getInstance().createRequestId('chat');
    this.activeRequestId = requestId;
    let generationFinished = false;
    let assistantRecord = new ChatMessageRecord();
    let failedMessage: ChatMessageView | undefined;
    try {
      const userRecord = await ChatRepository.getInstance().addMessage(
        this.currentSessionId, MessageRole.USER, displayInput
      );
      userRecord.rawContent = nativeInput;
      userRecord.displayContent = displayInput;
      await ChatRepository.getInstance().updateMessage(userRecord);
      await AttachmentRepository.getInstance().save(userRecord.id, attachments);
      const userMessage = new ChatMessageView(
        this.nextMessageId++, userRecord.id, userRecord.role, userRecord.displayContent, userRecord.state
      );
      userMessage.rawContent = nativeInput;
      userMessage.attachments = attachments;
      this.messages.push(userMessage);
      const historyRecords = await ChatRepository.getInstance().getMessages(this.currentSessionId);
      const request: NativeGenerateRequest = {
        requestId: requestId,
        messages: this.buildNativeHistory(historyRecords),
        maxTokens: this.runtimeSettings.maxTokens
      };
      assistantRecord = await ChatRepository.getInstance().addMessage(
        this.currentSessionId, MessageRole.ASSISTANT, '', GenerationState.GENERATING
      );
      failedMessage = new ChatMessageView(
        this.nextMessageId++, assistantRecord.id, assistantRecord.role, '', assistantRecord.state
      );
      const assistantMessage = failedMessage;
      this.messages.push(assistantMessage);
      this.messages = this.messages.slice();
      this.prompt = '';
      this.pendingAttachments = [];
      this.scrollToBottom();

      if (!this.nativeTarget) {
        throw new Error('模型尚未准备完成');
      }
      const generationPromise: Promise<NativeGenerateResult> = GenerationCoordinator.getInstance().generate(
        this.nativeTarget, request, (chunk: string) => {
        if (generationFinished || requestId !== this.activeRequestId) {
          return;
        }
        this.streamingAnswer += chunk;
        const parsed = ReasoningParser.parse(this.streamingAnswer);
        assistantMessage.rawContent = this.streamingAnswer;
        assistantMessage.reasoning = parsed.reasoning;
        assistantMessage.content = parsed.answer;
        assistantMessage.isThinking = parsed.isThinking;
        this.messages = this.messages.slice();
        if (this.followsLatest) {
          this.scrollToBottom();
        }
      });
      if (this.stopRequested) {
        GenerationCoordinator.getInstance().cancel(requestId);
      }
      const result = await generationPromise;
      generationFinished = true;
      if (requestId !== this.activeRequestId || result.requestId !== requestId) {
        return;
      }
      const rawResult = result.text || this.streamingAnswer;
      const parsedResult = ReasoningParser.parse(rawResult);
      // Display placeholder must never be persisted into the conversation history.
      const persistedAnswer = parsedResult.answer;
      assistantMessage.rawContent = rawResult;
      assistantMessage.reasoning = parsedResult.reasoning;
      assistantMessage.content = parsedResult.answer ||
        (result.canceled ? '已停止生成' : '模型未生成文本');
      assistantMessage.isThinking = false;
      assistantMessage.state = result.canceled ? GenerationState.CANCELED : GenerationState.COMPLETE;
      assistantMessage.metrics.loadMs = this.loadMs;
      assistantMessage.metrics.firstTokenMs = result.firstTokenMs;
      assistantMessage.metrics.promptTokens = result.promptTokens;
      assistantMessage.metrics.generatedTokens = result.generatedTokens;
      assistantMessage.metrics.prefillMs = result.prefillMs;
      assistantMessage.metrics.decodeMs = result.decodeMs;
      assistantMessage.metrics.totalMs = result.totalMs;
      assistantRecord.rawContent = assistantMessage.rawContent;
      assistantRecord.displayContent = persistedAnswer;
      assistantRecord.reasoningContent = assistantMessage.reasoning;
      assistantRecord.state = assistantMessage.state;
      assistantRecord.metrics = assistantMessage.metrics;
      await ChatRepository.getInstance().updateMessage(assistantRecord);
      const prefillSpeed = result.prefillMs > 0 ? result.promptTokens * 1000 / result.prefillMs : 0;
      const decodeSpeed = result.decodeMs > 0 ? result.generatedTokens * 1000 / result.decodeMs : 0;
      const state = result.canceled ? '已停止' : '完成';
      this.metrics = `${state} · 首字 ${(result.firstTokenMs / 1000).toFixed(2)}s · ` +
        `预填充 ${prefillSpeed.toFixed(1)} tok/s · 解码 ${decodeSpeed.toFixed(1)} tok/s`;
      this.messages = this.messages.slice();
      // The final content is usually taller than the last streamed snapshot, so scroll again
      // after layout instead of only relying on the per-chunk scroll.
      setTimeout(() => {
        if (this.followsLatest) {
          this.scrollToBottom();
        }
      }, 0);
    } catch (error) {
      generationFinished = true;
      if (requestId !== this.activeRequestId) {
        return;
      }
      const generationError = error as Error;
      this.error = generationError.message;
      if (assistantRecord.id) {
        const parsed = ReasoningParser.parse(this.streamingAnswer);
        assistantRecord.rawContent = this.streamingAnswer;
        assistantRecord.displayContent = parsed.answer || (parsed.reasoning ? '生成中断' : '生成失败');
        assistantRecord.reasoningContent = parsed.reasoning;
        assistantRecord.state = GenerationState.FAILED;
        if (failedMessage) {
          failedMessage.rawContent = assistantRecord.rawContent;
          failedMessage.reasoning = assistantRecord.reasoningContent;
          failedMessage.content = assistantRecord.displayContent;
          failedMessage.isThinking = false;
          failedMessage.state = GenerationState.FAILED;
          this.messages = this.messages.slice();
        }
        await ChatRepository.getInstance().updateMessage(assistantRecord);
      }
    } finally {
      this.isGenerating = false;
      if (this.activeRequestId === requestId) {
        this.activeRequestId = '';
        this.stopRequested = false;
      }
      if (this.pendingSessionId) {
        const pendingSessionId = this.pendingSessionId;
        this.pendingSessionId = '';
        this.isModelReady = false;
        this.restoreRequestedSession(pendingSessionId);
      } else if (this.modelSwitchPending) {
        this.modelSwitchPending = false;
        this.isModelReady = false;
        this.applyActiveModel().catch((error: Error) => {
          this.error = error.message;
          this.modelStatus = '模型加载失败';
        });
      } else if (this.settingsReloadPending) {
        this.settingsReloadPending = false;
        this.isModelReady = false;
        this.loadModel().catch((error: Error) => {
          this.error = error.message;
          this.modelStatus = '模型加载失败';
        });
      } else if (this.isModelReady) {
        this.modelStatus = '模型已就绪';
      }
    }
  }

  private scrollToBottom(): void {
    this.messageScroller.scrollEdge(Edge.Bottom);
    this.followsLatest = true;
  }

  private stopGeneration(): void {
    this.stopRequested = true;
    if (this.activeRequestId) {
      GenerationCoordinator.getInstance().cancel(this.activeRequestId);
    }
    this.modelStatus = '正在停止生成';
  }

  private startNewConversation(): void {
    if (this.isGenerating) {
      return;
    }
    this.createNewSession().catch((error: Error) => {
      this.error = error.message;
    });
  }

  private cloneRuntimeSettings(): ModelRuntimeSettings {
    const copy = new ModelRuntimeSettings();
    copy.backendType = this.runtimeSettings.backendType;
    copy.threadNum = this.runtimeSettings.threadNum;
    copy.precision = this.runtimeSettings.precision;
    copy.memory = this.runtimeSettings.memory;
    copy.useMmap = this.runtimeSettings.useMmap;
    copy.promptCache = this.runtimeSettings.promptCache;
    copy.maxTokens = this.runtimeSettings.maxTokens;
    copy.systemPrompt = this.runtimeSettings.systemPrompt;
    copy.thinking = this.runtimeSettings.thinking;
    copy.samplerType = this.runtimeSettings.samplerType;
    copy.temperature = this.runtimeSettings.temperature;
    copy.topP = this.runtimeSettings.topP;
    copy.topK = this.runtimeSettings.topK;
    copy.minP = this.runtimeSettings.minP;
    copy.tfsZ = this.runtimeSettings.tfsZ;
    copy.typical = this.runtimeSettings.typical;
    copy.penalty = this.runtimeSettings.penalty;
    copy.presencePenalty = this.runtimeSettings.presencePenalty;
    copy.frequencyPenalty = this.runtimeSettings.frequencyPenalty;
    copy.penaltyWindow = this.runtimeSettings.penaltyWindow;
    copy.nGram = this.runtimeSettings.nGram;
    copy.nGramFactor = this.runtimeSettings.nGramFactor;
    return copy;
  }

  private openModelPicker(): void {
    if (this.isGenerating || this.modelLoading) {
      return;
    }
    ModelRepository.getInstance().listModels().then((models: ModelRecord[]) => {
      const ready: ModelRecord[] = models.filter((model: ModelRecord) => model.state === ModelState.READY);
      if (ready.length === 0) {
        this.error = '没有可用的模型，请先在模型页导入或下载';
        return;
      }
      let controller: CustomDialogController | null = null;
      controller = new CustomDialogController({
        builder: ModelPickerSheet({
          models: ready,
          activeModelId: this.activeModelId,
          onPick: (model: ModelRecord) => {
            this.switchModel(model);
          }
        }),
        autoCancel: true,
        alignment: DialogAlignment.Bottom,
        customStyle: true
      });
      controller.open();
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private switchModel(model: ModelRecord): void {
    SettingsStore.getInstance().putString(ACTIVE_MODEL_ID_KEY, model.id).then(() => {
      if (this.activeModelId === model.id) {
        return;
      }
      this.modelStatus = '正在加载模型';
      this.activeModelId = model.id;
    }).catch((error: Error) => {
      this.error = error.message;
    });
  }

  private copyToClipboard(text: string): void {
    try {
      const data = pasteboard.createData(pasteboard.MIMETYPE_TEXT_PLAIN, text);
      pasteboard.getSystemPasteboard().setData(data).then(() => {
        this.modelStatus = '配置已复制到剪贴板';
      }).catch((error: Error) => {
        this.error = `复制失败：${error.message}`;
      });
    } catch (error) {
      this.error = `复制失败：${(error as Error).message}`;
    }
  }

  private showConfigInfo(): void {
    const settings = this.runtimeSettings;
    const info: string = [
      `model_id: ${this.loadedModelId || this.activeModelId}`,
      `config_path: ${this.nativeTarget?.configPath || ''}`,
      `backend_type: ${settings.backendType}`,
      `thread_num: ${settings.threadNum}`,
      `precision: ${settings.precision}`,
      `memory: ${settings.memory}`,
      `use_mmap: ${settings.useMmap}`,
      `prompt_cache: ${settings.promptCache}`,
      `sampler_type: ${settings.samplerType}`,
      `temperature: ${settings.temperature}`,
      `top_p: ${settings.topP}`,
      `top_k: ${settings.topK}`,
      `min_p: ${settings.minP}`,
      `tfs_z: ${settings.tfsZ}`,
      `typical: ${settings.typical}`,
      `repetition_penalty: ${settings.penalty}`,
      `presence_penalty: ${settings.presencePenalty}`,
      `frequency_penalty: ${settings.frequencyPenalty}`,
      `penalty_window: ${settings.penaltyWindow}`,
      `n_gram: ${settings.nGram}`,
      `ngram_factor: ${settings.nGramFactor}`,
      `max_new_tokens: ${settings.maxTokens}`,
      `enable_thinking: ${settings.thinking}`
    ].join('\n');
    let controller: CustomDialogController | null = null;
    controller = new CustomDialogController({
      builder: InfoSheet({
        title: '配置信息',
        message: info,
        onCopy: () => this.copyToClipboard(info)
      }),
      autoCancel: true,
      alignment: DialogAlignment.Bottom,
      customStyle: true
    });
    controller.open();
  }

  private toggleThinking(): void {
    if (!this.isModelReady || this.isGenerating) {
      return;
    }
    const next = this.cloneRuntimeSettings();
    next.thinking = !next.thinking;
    SettingsStore.getInstance().putModelRuntimeSettings(this.loadedModelId || this.activeModelId, next)
      .then((revision: number) => {
        this.settingsRevision = revision;
      })
      .catch((error: Error) => {
        this.error = error.message;
      });
  }

  private reportIssue(): void {
    const context = this.getUIContext().getHostContext() as common.UIAbilityContext;
    context.openLink('https://github.com/alibaba/MNN/issues').catch((error: Error) => {
      this.error = `无法打开浏览器：${error.message}`;
    });
  }

  build() {
    Column() {
      Row() {
        Column({ space: 4 }) {
          Text('MNN Chat')
            .fontSize(24)
            .fontWeight(FontWeight.Bold)
            .fontColor('#171A1F')
          Row({ space: 6 }) {
            Row()
              .width(7)
              .height(7)
              .borderRadius(4)
              .backgroundColor(this.isModelReady ? '#20A464' : '#F59E0B')
            Text(`${this.activeModelName || this.activeModelId} · ${this.modelStatus}`)
              .fontSize(13)
              .fontColor('#6B7280')
              .maxLines(1)
              .textOverflow({ overflow: TextOverflow.Ellipsis })
            Text('▾')
              .fontSize(12)
              .fontColor('#6B7280')
          }
          .onClick(() => this.openModelPicker())
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)

        Button('⋯')
          .fontSize(18)
          .fontColor('#171A1F')
          .backgroundColor(Color.Transparent)
          .height(38)
          .width(38)
          .bindMenu([
            {
              value: '配置信息',
              action: () => this.showConfigInfo()
            },
            {
              value: this.runtimeSettings.thinking ? '关闭思考模式' : '开启思考模式',
              action: () => this.toggleThinking()
            },
            {
              value: '上报问题',
              action: () => this.reportIssue()
            }
          ])

        Button('新对话')
          .fontSize(14)
          .fontColor('#2864DC')
          .backgroundColor('#EEF3FF')
          .height(38)
          .enabled(!this.isGenerating && this.initialized)
          .onClick(() => this.startNewConversation())
      }
      .width('100%')
      .padding({ left: 18, right: 18, top: 18, bottom: 14 })
      .backgroundColor(Color.White)

      Divider()
        .color('#E8EAED')

      List({ space: 16, scroller: this.messageScroller }) {
        ForEach(this.messages, (message: ChatMessageView) => {
          ListItem() {
            MessageBubble({ message: message })
          }
        }, (message: ChatMessageView) => message.id.toString())
      }
      .layoutWeight(1)
      .width('100%')
      .padding({ left: 16, right: 16, top: 18, bottom: 12 })
      .backgroundColor('#FAFBFC')
      .onScrollIndex((start: number, end: number) => {
        this.followsLatest = end >= this.messages.length - 1;
      })

      if (!this.followsLatest && this.messages.length > 0) {
        Button('回到底部')
          .height(32)
          .fontSize(12)
          .fontColor('#2864DC')
          .backgroundColor('#EEF3FF')
          .onClick(() => this.scrollToBottom())
      }

      if (this.metrics) {
        Text(this.metrics)
          .fontSize(12)
          .fontColor('#6B7280')
          .width('100%')
          .padding({ left: 18, right: 18, top: 8 })
      }

      if (this.error) {
        Text(`错误：${this.error}`)
          .fontSize(13)
          .fontColor('#C62828')
          .width('100%')
          .padding({ left: 18, right: 18, top: 8 })
      }

      Column({ space: 10 }) {
        Row({ space: 8 }) {
          Text('长文本测试')
            .fontSize(12)
            .fontColor('#7A7F87')
          ForEach([256, 512, 1024], (length: number) => {
            Button(`${length} 字符`)
              .height(30)
              .fontSize(12)
              .fontColor('#2864DC')
              .backgroundColor('#EEF3FF')
              .enabled(this.isModelReady && !this.isGenerating)
              .onClick(() => this.fillTestPrompt(length))
          })
        }
        .width('100%')

        AttachmentPicker({
          attachments: this.pendingAttachments,
          error: this.error,
          sessionId: this.currentSessionId,
          capabilities: this.modelCapabilities
        })

        TextArea({ placeholder: this.isModelReady ? '输入消息' : '模型加载完成后即可对话', text: this.prompt })
          .height(92)
          .fontSize(16)
          .backgroundColor('#F3F5F8')
          .borderRadius(14)
          .padding(12)
          .enabled(this.isModelReady && !this.isGenerating)
          .onChange((value: string) => {
            this.prompt = value;
          })

        Row({ space: 10 }) {
          Text(this.isGenerating ? '回答正在本机生成' : '内容仅在设备本地处理')
            .fontSize(12)
            .fontColor('#7A7F87')
            .layoutWeight(1)

          if (this.isGenerating) {
            Button('停止')
              .height(42)
              .fontSize(15)
              .fontColor('#C62828')
              .backgroundColor('#FDECEC')
              .onClick(() => this.stopGeneration())
          } else {
            Button('发送')
              .height(42)
              .fontSize(15)
              .fontColor(Color.White)
              .backgroundColor('#2864DC')
              .enabled(this.isModelReady && (this.prompt.trim().length > 0 || this.pendingAttachments.length > 0))
              .onClick(() => this.runInference())
          }
        }
        .width('100%')
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })
      .backgroundColor(Color.White)
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#FAFBFC')
  }
}

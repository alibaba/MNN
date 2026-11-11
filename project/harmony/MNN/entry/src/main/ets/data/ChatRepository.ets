import { tr } from '../services/I18n';
import relationalStore from '@ohos.data.relationalStore';
import fs from '@ohos.file.fs';
import { AppDatabase } from './AppDatabase';
import {
  ChatMessageRecord,
  ChatSessionRecord,
  GenerationMetrics,
  GenerationState,
  MessageRole
} from '../domain/AppTypes';

let idSequence: number = 0;

function createId(prefix: string): string {
  idSequence++;
  return `${prefix}-${Date.now()}-${idSequence}`;
}

export class ChatRepository {
  private static instance: ChatRepository = new ChatRepository();

  static getInstance(): ChatRepository {
    return ChatRepository.instance;
  }

  async createSession(modelId: string, title: string = tr('新对话')): Promise<ChatSessionRecord> {
    const now = Date.now();
    const session = new ChatSessionRecord();
    session.id = createId('session');
    session.title = title;
    session.modelId = modelId;
    session.createdAt = now;
    session.updatedAt = now;
    const values: relationalStore.ValuesBucket = {
      id: session.id,
      title: session.title,
      model_id: session.modelId,
      created_at: session.createdAt,
      updated_at: session.updatedAt
    };
    await AppDatabase.getInstance().getStore().insert('conversations', values);
    return session;
  }

  async getSession(sessionId: string): Promise<ChatSessionRecord | undefined> {
    const predicates = new relationalStore.RdbPredicates('conversations');
    predicates.equalTo('id', sessionId);
    const result = await AppDatabase.getInstance().getStore().query(predicates);
    try {
      if (!result.goToFirstRow()) {
        return undefined;
      }
      const session = new ChatSessionRecord();
      session.id = result.getString(result.getColumnIndex('id'));
      session.title = result.getString(result.getColumnIndex('title'));
      session.modelId = result.getString(result.getColumnIndex('model_id'));
      session.createdAt = result.getLong(result.getColumnIndex('created_at'));
      session.updatedAt = result.getLong(result.getColumnIndex('updated_at'));
      return session;
    } finally {
      result.close();
    }
  }

  async updateSessionModel(sessionId: string, modelId: string): Promise<void> {
    const values: relationalStore.ValuesBucket = {
      model_id: modelId,
      updated_at: Date.now()
    };
    const predicates = new relationalStore.RdbPredicates('conversations');
    predicates.equalTo('id', sessionId);
    await AppDatabase.getInstance().getStore().update(values, predicates);
  }

  async listSessions(): Promise<ChatSessionRecord[]> {
    const store = AppDatabase.getInstance().getStore();
    const predicates = new relationalStore.RdbPredicates('conversations');
    predicates.orderByDesc('updated_at');
    const result = await store.query(predicates);
    const sessions: ChatSessionRecord[] = [];
    try {
      while (result.goToNextRow()) {
        const session = new ChatSessionRecord();
        session.id = result.getString(result.getColumnIndex('id'));
        session.title = result.getString(result.getColumnIndex('title'));
        session.modelId = result.getString(result.getColumnIndex('model_id'));
        session.createdAt = result.getLong(result.getColumnIndex('created_at'));
        session.updatedAt = result.getLong(result.getColumnIndex('updated_at'));
        sessions.push(session);
      }
    } finally {
      result.close();
    }
    return sessions;
  }

  async getMessages(sessionId: string): Promise<ChatMessageRecord[]> {
    const store = AppDatabase.getInstance().getStore();
    const predicates = new relationalStore.RdbPredicates('messages');
    predicates.equalTo('session_id', sessionId).orderByAsc('created_at');
    const result = await store.query(predicates);
    const messages: ChatMessageRecord[] = [];
    try {
      while (result.goToNextRow()) {
        const message = new ChatMessageRecord();
        message.id = result.getString(result.getColumnIndex('id'));
        message.sessionId = result.getString(result.getColumnIndex('session_id'));
        message.role = result.getString(result.getColumnIndex('role')) as MessageRole;
        message.rawContent = result.getString(result.getColumnIndex('raw_content'));
        message.displayContent = result.getString(result.getColumnIndex('display_content'));
        message.reasoningContent = result.getString(result.getColumnIndex('reasoning_content'));
        message.state = result.getString(result.getColumnIndex('generation_state')) as GenerationState;
        message.createdAt = result.getLong(result.getColumnIndex('created_at'));
        message.metrics = this.parseMetrics(result.getString(result.getColumnIndex('metrics_json')));
        messages.push(message);
      }
    } finally {
      result.close();
    }
    return messages;
  }

  async addMessage(sessionId: string, role: MessageRole, content: string,
    state: GenerationState = GenerationState.COMPLETE): Promise<ChatMessageRecord> {
    const message = new ChatMessageRecord();
    message.id = createId('message');
    message.sessionId = sessionId;
    message.role = role;
    message.rawContent = content;
    message.displayContent = content;
    message.state = state;
    message.createdAt = Date.now();
    const values: relationalStore.ValuesBucket = this.messageValues(message);
    await AppDatabase.getInstance().getStore().insert('messages', values);
    await this.touchSession(sessionId, message.createdAt);
    if (role === MessageRole.USER) {
      await this.useFirstMessageAsTitle(sessionId, content);
    }
    return message;
  }

  async updateMessage(message: ChatMessageRecord): Promise<void> {
    const predicates = new relationalStore.RdbPredicates('messages');
    predicates.equalTo('id', message.id);
    await AppDatabase.getInstance().getStore().update(this.messageValues(message), predicates);
    await this.touchSession(message.sessionId, Date.now());
  }

  async renameSession(sessionId: string, title: string): Promise<void> {
    const values: relationalStore.ValuesBucket = {
      title: title.trim(),
      updated_at: Date.now()
    };
    const predicates = new relationalStore.RdbPredicates('conversations');
    predicates.equalTo('id', sessionId);
    await AppDatabase.getInstance().getStore().update(values, predicates);
  }

  async recoverInterruptedGenerations(): Promise<void> {
    const predicates = new relationalStore.RdbPredicates('messages');
    predicates.equalTo('generation_state', GenerationState.GENERATING);
    const result = await AppDatabase.getInstance().getStore().query(predicates);
    const interrupted: ChatMessageRecord[] = [];
    try {
      while (result.goToNextRow()) {
        const message = new ChatMessageRecord();
        message.id = result.getString(result.getColumnIndex('id'));
        message.sessionId = result.getString(result.getColumnIndex('session_id'));
        message.role = result.getString(result.getColumnIndex('role')) as MessageRole;
        message.rawContent = result.getString(result.getColumnIndex('raw_content'));
        message.displayContent = result.getString(result.getColumnIndex('display_content'));
        message.reasoningContent = result.getString(result.getColumnIndex('reasoning_content'));
        message.createdAt = result.getLong(result.getColumnIndex('created_at'));
        message.metrics = this.parseMetrics(result.getString(result.getColumnIndex('metrics_json')));
        message.state = GenerationState.CANCELED;
        if (!message.displayContent) {
          message.displayContent = tr('上次生成意外中断');
          message.rawContent = message.displayContent;
        }
        interrupted.push(message);
      }
    } finally {
      result.close();
    }
    for (let index = 0; index < interrupted.length; index++) {
      await this.updateMessage(interrupted[index]);
    }
  }

  async deleteSession(sessionId: string, filesDir: string): Promise<void> {
    const store = AppDatabase.getInstance().getStore();
    const messagePredicates = new relationalStore.RdbPredicates('messages');
    messagePredicates.equalTo('session_id', sessionId);
    const messageResult = await store.query(messagePredicates, ['id']);
    const messageIds: string[] = [];
    try {
      while (messageResult.goToNextRow()) {
        messageIds.push(messageResult.getString(messageResult.getColumnIndex('id')));
      }
    } finally {
      messageResult.close();
    }
    for (let index = 0; index < messageIds.length; index++) {
      const attachmentPredicates = new relationalStore.RdbPredicates('attachments');
      attachmentPredicates.equalTo('message_id', messageIds[index]);
      await store.delete(attachmentPredicates);
    }
    await store.delete(messagePredicates);
    const sessionPredicates = new relationalStore.RdbPredicates('conversations');
    sessionPredicates.equalTo('id', sessionId);
    await store.delete(sessionPredicates);
    const safeSessionId = sessionId.replace(/[^a-zA-Z0-9._-]/g, '-').substring(0, 96);
    const historyPath = `${filesDir}/history/${safeSessionId}`;
    if (fs.accessSync(historyPath) && fs.lstatSync(historyPath).isDirectory() &&
      !fs.lstatSync(historyPath).isSymbolicLink()) {
      fs.rmdirSync(historyPath);
    }
  }

  private messageValues(message: ChatMessageRecord): relationalStore.ValuesBucket {
    return {
      id: message.id,
      session_id: message.sessionId,
      role: message.role,
      raw_content: message.rawContent,
      display_content: message.displayContent,
      reasoning_content: message.reasoningContent,
      generation_state: message.state,
      created_at: message.createdAt,
      metrics_json: JSON.stringify(message.metrics)
    };
  }

  private async touchSession(sessionId: string, updatedAt: number): Promise<void> {
    const values: relationalStore.ValuesBucket = { updated_at: updatedAt };
    const predicates = new relationalStore.RdbPredicates('conversations');
    predicates.equalTo('id', sessionId);
    await AppDatabase.getInstance().getStore().update(values, predicates);
  }

  private async useFirstMessageAsTitle(sessionId: string, content: string): Promise<void> {
    const messages = await this.getMessages(sessionId);
    let userCount = 0;
    messages.forEach((message: ChatMessageRecord) => {
      if (message.role === MessageRole.USER) {
        userCount++;
      }
    });
    if (userCount !== 1) {
      return;
    }
    const normalized = content.replace(/\s+/g, ' ').trim();
    const title = normalized.length > 24 ? `${normalized.substring(0, 24)}…` : normalized;
    if (title) {
      await this.renameSession(sessionId, title);
    }
  }

  private parseMetrics(value: string): GenerationMetrics {
    const metrics = new GenerationMetrics();
    if (!value) {
      return metrics;
    }
    const parsed = JSON.parse(value) as GenerationMetrics;
    metrics.loadMs = parsed.loadMs;
    metrics.firstTokenMs = parsed.firstTokenMs;
    metrics.promptTokens = parsed.promptTokens;
    metrics.generatedTokens = parsed.generatedTokens;
    metrics.prefillMs = parsed.prefillMs;
    metrics.decodeMs = parsed.decodeMs;
    metrics.totalMs = parsed.totalMs;
    return metrics;
  }
}

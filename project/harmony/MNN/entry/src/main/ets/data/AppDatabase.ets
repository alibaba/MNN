import { common } from '@kit.AbilityKit';
import relationalStore from '@ohos.data.relationalStore';

const DATABASE_NAME: string = 'mnn_chat.db';
const DATABASE_VERSION: number = 1;

export class AppDatabase {
  private static instance: AppDatabase = new AppDatabase();
  private store?: relationalStore.RdbStore;

  static getInstance(): AppDatabase {
    return AppDatabase.instance;
  }

  async initialize(context: common.UIAbilityContext): Promise<void> {
    if (this.store) {
      return;
    }
    const config: relationalStore.StoreConfig = {
      name: DATABASE_NAME,
      securityLevel: relationalStore.SecurityLevel.S1
    };
    const store = await relationalStore.getRdbStore(context, config);
    await this.createSchema(store);
    this.store = store;
  }

  getStore(): relationalStore.RdbStore {
    if (!this.store) {
      throw new Error('应用数据库尚未初始化');
    }
    return this.store;
  }

  private async createSchema(store: relationalStore.RdbStore): Promise<void> {
    await store.executeSql(
      'CREATE TABLE IF NOT EXISTS conversations (' +
      'id TEXT PRIMARY KEY, title TEXT NOT NULL, model_id TEXT NOT NULL, ' +
      'created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)'
    );
    await store.executeSql(
      'CREATE TABLE IF NOT EXISTS messages (' +
      'id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, ' +
      'raw_content TEXT NOT NULL, display_content TEXT NOT NULL, reasoning_content TEXT NOT NULL, ' +
      'generation_state TEXT NOT NULL, created_at INTEGER NOT NULL, metrics_json TEXT NOT NULL)'
    );
    await store.executeSql(
      'CREATE TABLE IF NOT EXISTS attachments (' +
      'id TEXT PRIMARY KEY, message_id TEXT NOT NULL, media_type TEXT NOT NULL, local_path TEXT NOT NULL, ' +
      'display_name TEXT NOT NULL, size_bytes INTEGER NOT NULL, created_at INTEGER NOT NULL)'
    );
    await store.executeSql(
      'CREATE TABLE IF NOT EXISTS models (' +
      'id TEXT PRIMARY KEY, name TEXT NOT NULL, config_path TEXT NOT NULL UNIQUE, root_path TEXT NOT NULL, ' +
      'state TEXT NOT NULL, is_legacy INTEGER NOT NULL, is_protected INTEGER NOT NULL, is_pinned INTEGER NOT NULL, ' +
      'last_used_at INTEGER NOT NULL, created_at INTEGER NOT NULL)'
    );
    await store.executeSql(
      'CREATE TABLE IF NOT EXISTS download_tasks (' +
      'id TEXT PRIMARY KEY, model_id TEXT NOT NULL, provider TEXT NOT NULL, source_url TEXT NOT NULL, ' +
      'target_path TEXT NOT NULL, state TEXT NOT NULL, downloaded_bytes INTEGER NOT NULL, ' +
      'total_bytes INTEGER NOT NULL, updated_at INTEGER NOT NULL)'
    );
    await store.executeSql(
      'CREATE TABLE IF NOT EXISTS benchmark_results (' +
      'id TEXT PRIMARY KEY, model_id TEXT NOT NULL, scenario TEXT NOT NULL, config_snapshot TEXT NOT NULL, ' +
      'result_json TEXT NOT NULL, created_at INTEGER NOT NULL)'
    );
    await store.executeSql('CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at)');
    await store.executeSql('CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at)');
    await store.executeSql('CREATE INDEX IF NOT EXISTS idx_models_last_used ON models(last_used_at)');
    store.version = DATABASE_VERSION;
  }
}

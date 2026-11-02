import relationalStore from '@ohos.data.relationalStore';
import { AppDatabase } from './AppDatabase';
import { DownloadTaskRecord } from '../domain/AppTypes';

export class DownloadRepository {
  private static instance: DownloadRepository = new DownloadRepository();

  static getInstance(): DownloadRepository {
    return DownloadRepository.instance;
  }

  async save(task: DownloadTaskRecord): Promise<void> {
    const store = AppDatabase.getInstance().getStore();
    const predicates = new relationalStore.RdbPredicates('download_tasks');
    predicates.equalTo('id', task.id);
    const result = await store.query(predicates);
    let exists = false;
    try {
      exists = result.goToFirstRow();
    } finally {
      result.close();
    }
    const values = this.values(task);
    if (exists) {
      const updatePredicates = new relationalStore.RdbPredicates('download_tasks');
      updatePredicates.equalTo('id', task.id);
      await store.update(values, updatePredicates);
    } else {
      await store.insert('download_tasks', values);
    }
  }

  async list(modelId?: string): Promise<DownloadTaskRecord[]> {
    const predicates = new relationalStore.RdbPredicates('download_tasks');
    if (modelId) {
      predicates.equalTo('model_id', modelId);
    }
    predicates.orderByDesc('updated_at');
    const result = await AppDatabase.getInstance().getStore().query(predicates);
    const tasks: DownloadTaskRecord[] = [];
    try {
      while (result.goToNextRow()) {
        const task = new DownloadTaskRecord();
        task.id = result.getString(result.getColumnIndex('id'));
        task.modelId = result.getString(result.getColumnIndex('model_id'));
        task.provider = result.getString(result.getColumnIndex('provider'));
        task.sourceUrl = result.getString(result.getColumnIndex('source_url'));
        task.targetPath = result.getString(result.getColumnIndex('target_path'));
        task.state = result.getString(result.getColumnIndex('state'));
        task.downloadedBytes = result.getLong(result.getColumnIndex('downloaded_bytes'));
        task.totalBytes = result.getLong(result.getColumnIndex('total_bytes'));
        task.updatedAt = result.getLong(result.getColumnIndex('updated_at'));
        tasks.push(task);
      }
    } finally {
      result.close();
    }
    return tasks;
  }

  async removeModel(modelId: string): Promise<void> {
    const predicates = new relationalStore.RdbPredicates('download_tasks');
    predicates.equalTo('model_id', modelId);
    await AppDatabase.getInstance().getStore().delete(predicates);
  }

  private values(task: DownloadTaskRecord): relationalStore.ValuesBucket {
    return {
      id: task.id,
      model_id: task.modelId,
      provider: task.provider,
      source_url: task.sourceUrl,
      target_path: task.targetPath,
      state: task.state,
      downloaded_bytes: task.downloadedBytes,
      total_bytes: task.totalBytes,
      updated_at: task.updatedAt
    };
  }
}

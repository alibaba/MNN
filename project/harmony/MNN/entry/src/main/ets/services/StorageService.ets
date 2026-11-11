import { common } from '@kit.AbilityKit';
import fs from '@ohos.file.fs';
import statvfs from '@ohos.file.statvfs';
import { ModelRepository } from '../data/ModelRepository';
import { ModelRecord } from '../domain/AppTypes';

export class StorageEntry {
  modelId: string = '';
  modelName: string = '';
  modelBytes: number = 0;
  cacheBytes: number = 0;
  cacheDir: string = '';
  orphan: boolean = false;
}

export class StorageSnapshot {
  totalBytes: number = 0;
  freeBytes: number = 0;
  modelBytes: number = 0;
  cacheBytes: number = 0;
  downloadBytes: number = 0;
  orphanBytes: number = 0;
  entries: StorageEntry[] = [];
}

/**
 * Groups app storage use by model, mirroring the Android client's storage management screen:
 * filesystem total/free, model files, per-model runtime (mmap/prompt) cache and orphan cache.
 */
export class StorageService {
  private static instance: StorageService = new StorageService();

  static getInstance(): StorageService {
    return StorageService.instance;
  }

  async snapshot(context: common.UIAbilityContext): Promise<StorageSnapshot> {
    const snapshot = new StorageSnapshot();
    try {
      snapshot.totalBytes = statvfs.getTotalSizeSync(context.filesDir);
      snapshot.freeBytes = statvfs.getFreeSizeSync(context.filesDir);
    } catch (error) {
    }
    const models = await ModelRepository.getInstance().listModels();
    const entries: StorageEntry[] = [];
    models.forEach((model: ModelRecord) => {
      const entry = new StorageEntry();
      entry.modelId = model.id;
      entry.modelName = model.name || model.id;
      entry.modelBytes = model.sizeBytes;
      entry.cacheDir = `${context.cacheDir}/llm-${this.safeId(model.id)}`;
      entry.cacheBytes = this.directorySize(entry.cacheDir);
      snapshot.modelBytes += entry.modelBytes;
      entries.push(entry);
    });
    const known = entries.map((entry: StorageEntry) => this.safeId(entry.modelId));
    this.listDirectories(context.cacheDir).forEach((name: string) => {
      if (!name.startsWith('llm-')) {
        return;
      }
      const id = name.substring(4);
      if (known.indexOf(id) >= 0) {
        return;
      }
      const entry = new StorageEntry();
      entry.modelId = id;
      entry.modelName = id;
      entry.orphan = true;
      entry.cacheDir = `${context.cacheDir}/${name}`;
      entry.cacheBytes = this.directorySize(entry.cacheDir);
      snapshot.orphanBytes += entry.cacheBytes;
      entries.push(entry);
    });
    snapshot.downloadBytes = this.directorySize(`${context.filesDir}/.mnn-downloads`);
    entries.forEach((entry: StorageEntry) => {
      snapshot.cacheBytes += entry.cacheBytes;
    });
    entries.sort((left: StorageEntry, right: StorageEntry) => {
      return (right.modelBytes + right.cacheBytes) - (left.modelBytes + left.cacheBytes);
    });
    snapshot.entries = entries;
    return snapshot;
  }

  clearModelCache(context: common.UIAbilityContext, entry: StorageEntry): number {
    const freed = entry.cacheBytes;
    this.removeDirectory(entry.cacheDir);
    return freed;
  }

  clearOrphanCaches(context: common.UIAbilityContext, snapshot: StorageSnapshot): number {
    let freed = 0;
    snapshot.entries.forEach((entry: StorageEntry) => {
      if (entry.orphan) {
        freed += entry.cacheBytes;
        this.removeDirectory(entry.cacheDir);
      }
    });
    return freed;
  }

  clearAllCaches(context: common.UIAbilityContext): number {
    let freed = 0;
    this.listEntries(context.cacheDir).forEach((name: string) => {
      const path = `${context.cacheDir}/${name}`;
      freed += this.directorySize(path);
      this.removeDirectory(path);
    });
    return freed;
  }

  private safeId(modelId: string): string {
    return modelId.replace(/[^a-zA-Z0-9._-]/g, '-');
  }

  private listEntries(directory: string): string[] {
    try {
      return fs.listFileSync(directory);
    } catch (error) {
      return [];
    }
  }

  private listDirectories(directory: string): string[] {
    const directories: string[] = [];
    this.listEntries(directory).forEach((name: string) => {
      try {
        if (fs.statSync(`${directory}/${name}`).isDirectory()) {
          directories.push(name);
        }
      } catch (error) {
      }
    });
    return directories;
  }

  private directorySize(path: string): number {
    try {
      const stat = fs.statSync(path);
      if (!stat.isDirectory()) {
        return stat.size;
      }
      let total = 0;
      fs.listFileSync(path).forEach((name: string) => {
        total += this.directorySize(`${path}/${name}`);
      });
      return total;
    } catch (error) {
      return 0;
    }
  }

  private removeDirectory(path: string): void {
    try {
      if (!fs.accessSync(path)) {
        return;
      }
      if (fs.statSync(path).isDirectory()) {
        fs.rmdirSync(path);
      } else {
        fs.unlinkSync(path);
      }
    } catch (error) {
    }
  }
}
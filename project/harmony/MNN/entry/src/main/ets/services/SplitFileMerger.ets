import fs from '@ohos.file.fs';

interface SplitChunkDocument {
  chunkIndex: number;
  chunkFileName: string;
  chunkSize: number;
}

interface SplitDocument {
  originalFileName: string;
  originalFileSize: number;
  totalChunks: number;
  chunks: SplitChunkDocument[];
}

class MergeGroup {
  name: string = '';
  expectedSize: number = 0;
  chunks: SplitChunkDocument[] = [];
}

const SPLITS_INFO_FILE: string = 'splits_info.json';
const COPY_BUFFER_BYTES: number = 1024 * 1024;

/**
 * Large models are sometimes published as <file>.partN chunks plus a splits_info.json descriptor.
 * Android merges them before loading; this mirrors that behavior so split models are usable.
 */
export class SplitFileMerger {
  static hasSplits(modelDir: string): boolean {
    return fs.accessSync(`${modelDir}/${SPLITS_INFO_FILE}`);
  }

  /** Whether a model artifact exists only as a set of split chunks (not merged yet). */
  static coversFile(modelDir: string, relativePath: string): boolean {
    const document = SplitFileMerger.readDocument(modelDir);
    if (!document) {
      return false;
    }
    const target = SplitFileMerger.baseName(relativePath);
    return document.chunks.some((chunk: SplitChunkDocument) =>
      SplitFileMerger.originalName(chunk.chunkFileName) === target);
  }

  /** Merges every split group in the directory. Returns true when a merge was performed. */
  static async mergeIfNeeded(modelDir: string): Promise<boolean> {
    if (!SplitFileMerger.hasSplits(modelDir)) {
      return false;
    }
    const document = SplitFileMerger.readDocument(modelDir);
    if (!document || document.chunks.length === 0) {
      throw new Error('splits_info.json 内容无效');
    }
    document.chunks.forEach((chunk: SplitChunkDocument) => {
      if (!SplitFileMerger.isSafeName(chunk.chunkFileName)) {
        throw new Error(`分片文件名不安全：${chunk.chunkFileName}`);
      }
    });

    const groups = SplitFileMerger.groupChunks(document);
    for (let index = 0; index < groups.length; index++) {
      SplitFileMerger.mergeGroup(modelDir, groups[index]);
    }
    fs.unlinkSync(`${modelDir}/${SPLITS_INFO_FILE}`);
    return true;
  }

  private static groupChunks(document: SplitDocument): MergeGroup[] {
    const groups: MergeGroup[] = [];
    document.chunks.forEach((chunk: SplitChunkDocument) => {
      const name = SplitFileMerger.originalName(chunk.chunkFileName);
      let group = groups.find((item: MergeGroup) => item.name === name);
      if (!group) {
        group = new MergeGroup();
        group.name = name;
        groups.push(group);
      }
      group.chunks.push(chunk);
      if (chunk.chunkSize > 0) {
        group.expectedSize += chunk.chunkSize;
      }
    });
    if (groups.length === 1 && groups[0].expectedSize <= 0) {
      groups[0].expectedSize = document.originalFileSize;
    }
    return groups;
  }

  private static mergeGroup(modelDir: string, group: MergeGroup): void {
    const outputPath = `${modelDir}/${group.name}`;
    if (fs.accessSync(outputPath)) {
      const size = fs.statSync(outputPath).size;
      if (group.expectedSize <= 0 || size === group.expectedSize) {
        SplitFileMerger.deleteChunks(modelDir, group);
        return;
      }
      fs.unlinkSync(outputPath);
    }

    const chunks = group.chunks.slice();
    chunks.sort((left: SplitChunkDocument, right: SplitChunkDocument) =>
      (left.chunkIndex || 0) - (right.chunkIndex || 0));

    const output = fs.openSync(outputPath,
      fs.OpenMode.CREATE | fs.OpenMode.WRITE_ONLY | fs.OpenMode.TRUNC);
    let written = 0;
    try {
      for (let index = 0; index < chunks.length; index++) {
        written += SplitFileMerger.appendChunk(`${modelDir}/${chunks[index].chunkFileName}`, output.fd);
      }
      fs.fsyncSync(output.fd);
    } finally {
      fs.closeSync(output);
    }
    const mergedSize = fs.statSync(outputPath).size;
    if ((group.expectedSize > 0 && mergedSize !== group.expectedSize) || mergedSize !== written) {
      fs.unlinkSync(outputPath);
      throw new Error(`合并 ${group.name} 失败：大小 ${mergedSize}，预期 ${group.expectedSize || written}`);
    }
    SplitFileMerger.deleteChunks(modelDir, group);
  }

  private static appendChunk(chunkPath: string, outputFd: number): number {
    const input = fs.openSync(chunkPath, fs.OpenMode.READ_ONLY);
    let total = 0;
    try {
      const buffer = new ArrayBuffer(COPY_BUFFER_BYTES);
      let readBytes = fs.readSync(input.fd, buffer);
      while (readBytes > 0) {
        fs.writeSync(outputFd, buffer, { length: readBytes });
        total += readBytes;
        readBytes = fs.readSync(input.fd, buffer);
      }
    } finally {
      fs.closeSync(input);
    }
    return total;
  }

  private static deleteChunks(modelDir: string, group: MergeGroup): void {
    group.chunks.forEach((chunk: SplitChunkDocument) => {
      const path = `${modelDir}/${chunk.chunkFileName}`;
      if (fs.accessSync(path)) {
        try {
          fs.unlinkSync(path);
        } catch (error) {
          // Leaving a stale chunk behind is harmless; the merged file is authoritative.
        }
      }
    });
  }

  private static readDocument(modelDir: string): SplitDocument | null {
    const path = `${modelDir}/${SPLITS_INFO_FILE}`;
    if (!fs.accessSync(path)) {
      return null;
    }
    try {
      return JSON.parse(fs.readTextSync(path)) as SplitDocument;
    } catch (error) {
      return null;
    }
  }

  private static originalName(chunkName: string): string {
    return chunkName.replace(/\.part\d+$/, '');
  }

  private static baseName(path: string): string {
    const normalized = path.startsWith('./') ? path.substring(2) : path;
    const index = normalized.lastIndexOf('/');
    return index >= 0 ? normalized.substring(index + 1) : normalized;
  }

  private static isSafeName(name: string): boolean {
    return name.length > 0 && name.indexOf('/') < 0 && name.indexOf('\\') < 0 &&
      name.indexOf('..') < 0 && name !== SPLITS_INFO_FILE;
  }
}
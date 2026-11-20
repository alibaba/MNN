import { hidebug } from '@kit.PerformanceAnalysisKit';
import { fileIo } from '@kit.CoreFileKit';
import { util } from '@kit.ArkTS';

interface RuntimeConfigProbe {
  use_mmap?: boolean;
}

class RollupFields {
  rssKb: number = -1;
  pssKb: number = -1;
  anonKb: number = -1;
  fileKb: number = -1;
}

/**
 * Logs process memory composition (anonymous vs file-backed) around model load and generation so
 * the effect of `use_mmap` can be observed in hilog with the [MemProbe] tag.
 */
export class MemoryProbe {
  static snapshot(tag: string, runtimeConfigJson: string = ''): void {
    let mmap = 'unknown';
    if (runtimeConfigJson) {
      try {
        const config = JSON.parse(runtimeConfigJson) as RuntimeConfigProbe;
        mmap = config.use_mmap === true ? 'on' : 'off';
      } catch (error) {
        mmap = 'parse-error';
      }
    }
    try {
      const pssKb = Number(hidebug.getPss());
      const rollup = MemoryProbe.readRollup();
      console.info(`[MemProbe] ${tag} mmap=${mmap} pss=${MemoryProbe.mb(pssKb)} rss=${MemoryProbe.mb(rollup.rssKb)} ` +
        `anon=${MemoryProbe.mb(rollup.anonKb)} file=${MemoryProbe.mb(rollup.fileKb)}`);
    } catch (error) {
      console.info(`[MemProbe] ${tag} mmap=${mmap} failed:${(error as Error).message}`);
    }
  }

  private static mb(kb: number): string {
    return kb < 0 ? 'n/a' : `${(kb / 1024).toFixed(1)}MB`;
  }

  private static readRollup(): RollupFields {
    const fields = new RollupFields();
    const file = fileIo.openSync('/proc/self/smaps_rollup', fileIo.OpenMode.READ_ONLY);
    try {
      // /proc files can return short reads, so accumulate until EOF.
      const buffer = new ArrayBuffer(16 * 1024);
      const chunks: Uint8Array[] = [];
      let total = 0;
      while (true) {
        const read = fileIo.readSync(file.fd, buffer);
        if (read <= 0) {
          break;
        }
        chunks.push(new Uint8Array(buffer.slice(0, read)));
        total += read;
      }
      const merged = new Uint8Array(total);
      let offset = 0;
      chunks.forEach((chunk: Uint8Array) => {
        merged.set(chunk, offset);
        offset += chunk.length;
      });
      const text = util.TextDecoder.create('utf-8').decodeToString(merged);
      text.split('\n').forEach((line: string) => {
        if (line.startsWith('Rss:')) {
          fields.rssKb = MemoryProbe.parseKb(line);
        } else if (line.startsWith('Pss:')) {
          fields.pssKb = MemoryProbe.parseKb(line);
        } else if (line.startsWith('RssAnon:')) {
          fields.anonKb = MemoryProbe.parseKb(line);
        } else if (line.startsWith('RssFile:')) {
          fields.fileKb = MemoryProbe.parseKb(line);
        }
      });
    } finally {
      fileIo.closeSync(file);
    }
    return fields;
  }

  private static parseKb(line: string): number {
    const parts = line.split(/\s+/);
    const value = parts.length > 1 ? Number(parts[1]) : NaN;
    return Number.isFinite(value) ? value : -1;
  }
}
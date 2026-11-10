import { media } from '@kit.MediaKit';
import { fileIo } from '@kit.CoreFileKit';

/** Minimal AVPlayer wrapper used to replay generated or recorded speech files. */
export class AudioPlaybackService {
  private static instance: AudioPlaybackService = new AudioPlaybackService();
  private player?: media.AVPlayer;
  private file?: fileIo.File;

  static getInstance(): AudioPlaybackService {
    return AudioPlaybackService.instance;
  }

  async play(path: string): Promise<void> {
    await this.stop();
    // AVPlayer only accepts fd:// or network sources, so hand it an open descriptor.
    const file = fileIo.openSync(path, fileIo.OpenMode.READ_ONLY);
    const size = fileIo.statSync(path).size;
    const player = await media.createAVPlayer();
    this.player = player;
    this.file = file;
    player.on('stateChange', (state: media.AVPlayerState) => {
      if (state === 'initialized') {
        player.prepare().catch(() => {
        });
      } else if (state === 'prepared') {
        player.play().catch(() => {
        });
      } else if (state === 'completed') {
        this.releasePlayer(player);
      }
    });
    player.on('error', () => {
      this.releasePlayer(player);
    });
    player.fdSrc = { fd: file.fd, offset: 0, length: size };
  }

  async stop(): Promise<void> {
    const player = this.player;
    this.player = undefined;
    if (player) {
      try {
        await player.release();
      } catch (error) {
      }
    }
    this.closeFile();
  }

  private releasePlayer(player: media.AVPlayer): void {
    player.release().catch(() => {
    });
    if (this.player === player) {
      this.player = undefined;
      this.closeFile();
    }
  }

  private closeFile(): void {
    if (this.file) {
      try {
        fileIo.closeSync(this.file);
      } catch (error) {
      }
      this.file = undefined;
    }
  }
}
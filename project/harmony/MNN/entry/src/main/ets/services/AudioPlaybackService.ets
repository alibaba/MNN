import { media } from '@kit.MediaKit';
import { BusinessError } from '@kit.BasicServicesKit';

/** Minimal AVPlayer wrapper used to replay generated or recorded speech files. */
export class AudioPlaybackService {
  private static instance: AudioPlaybackService = new AudioPlaybackService();
  private player?: media.AVPlayer;

  static getInstance(): AudioPlaybackService {
    return AudioPlaybackService.instance;
  }

  async play(path: string): Promise<void> {
    await this.stop();
    const player = await media.createAVPlayer();
    this.player = player;
    player.on('stateChange', (state: media.AVPlayerState) => {
      if (state === 'initialized') {
        player.prepare().catch(() => {
        });
      } else if (state === 'prepared') {
        player.play().catch(() => {
        });
      } else if (state === 'completed') {
        player.release().catch(() => {
        });
        if (this.player === player) {
          this.player = undefined;
        }
      }
    });
    player.on('error', (error: BusinessError) => {
      player.release().catch(() => {
      });
      if (this.player === player) {
        this.player = undefined;
      }
    });
    player.url = `file://${path}`;
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
  }
}
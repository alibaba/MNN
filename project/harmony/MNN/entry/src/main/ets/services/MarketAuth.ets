import { SettingsStore } from './SettingsStore';

/**
 * Stores the access token used for private model repositories on the supported hosting platforms.
 * Tokens are kept in the on-device preferences store only and never leave the device except as an
 * Authorization header for the corresponding provider.
 */
export class MarketAuth {
  static key(provider: string): string {
    return `market_token_${provider.toLowerCase()}`;
  }

  static async getToken(provider: string): Promise<string> {
    return await SettingsStore.getInstance().getString(MarketAuth.key(provider));
  }

  static async saveToken(provider: string, token: string): Promise<void> {
    await SettingsStore.getInstance().putString(MarketAuth.key(provider), token.trim());
  }
}
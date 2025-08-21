# Release Script Documentation

This directory contains scripts for building and releasing the MnnLlmChat Android application.

## Files

- `release.sh` - Main release script for building and publishing the app
- `release.config.example` - Example configuration file
- `setup_release.sh` - Setup script to help configure the release environment
- `test_build.sh` - Test script to verify build commands work correctly
- `get_repo_info.py` - Utility script for getting repository information

## Quick Start

1. **Setup the release environment**:
   ```bash
   ./scripts/setup_release.sh
   ```

2. **Edit the configuration**:
   ```bash
   # Edit the configuration file with your actual values
   nano scripts/release.config
   ```

3. **Test the build commands**:
   ```bash
   ./scripts/test_build.sh
   ```

4. **Run the release process**:
   ```bash
   source scripts/release.config
   ./scripts/release.sh
   ```

## Release Script Usage

The `release.sh` script automates the process of building and publishing both the standard flavor debug version (for CDN) and the Google Play flavor release version (for Google Play Store).

### Prerequisites

1. **Java 11 or higher** - Required for building Android projects
2. **Gradle Wrapper** - Should be present in the project root
3. **Signing Configuration** - Required for Google Play releases
4. **CDN Configuration** - Required for CDN uploads
5. **Google Play Configuration** - Required for Google Play uploads

### Configuration

1. Copy the example configuration file:
   ```bash
   cp scripts/release.config.example scripts/release.config
   ```

2. Edit `scripts/release.config` with your actual values:
   - **Signing Configuration**: Path to keystore file and credentials
   - **CDN Configuration**: Aliyun OSS endpoint and credentials
   - **Google Play Configuration**: Service account JSON file path

3. Source the configuration file before running the script:
   ```bash
   source scripts/release.config
   ```

### Running the Release Script

From the project root directory:

```bash
# Run the complete release process
./scripts/release.sh
```

### What the Script Does

1. **Requirements Check**: Verifies Java version, Gradle wrapper, and configuration
2. **Clean Build**: Removes previous builds and creates output directories
3. **Build Standard Debug**: Builds the standard flavor debug APK for CDN distribution
4. **Build Google Play Release**: Builds the Google Play flavor release APK/AAB for Google Play Store
5. **Upload to CDN**: Uploads the standard debug APK to Aliyun OSS CDN
6. **Upload to Google Play**: Uploads the Google Play release APK/AAB to Google Play Store
7. **Generate Release Notes**: Creates release notes with build information

### Output Structure

```
release_outputs/
├── cdn/
│   └── app-standard-debug.apk
├── googleplay/
│   ├── app-googleplay-release.apk
│   └── app-googleplay-release.aab
└── release_notes.md
```

### Environment Variables

The script uses the following environment variables (can be set in `release.config`):

#### Signing Configuration
- `KEYSTORE_FILE` - Path to keystore file
- `KEYSTORE_PASSWORD` - Keystore password
- `KEY_ALIAS` - Key alias
- `KEY_PASSWORD` - Key password

#### CDN Configuration
- `CDN_ENDPOINT` - Aliyun OSS endpoint
- `CDN_ACCESS_KEY` - Aliyun OSS access key ID
- `CDN_SECRET_KEY` - Aliyun OSS access key secret
- `CDN_BUCKET` - Aliyun OSS bucket name

#### Google Play Configuration
- `GOOGLE_PLAY_SERVICE_ACCOUNT` - Path to Google Play service account JSON file
- `GOOGLE_PLAY_PACKAGE_NAME` - Google Play package name

### Troubleshooting

#### Common Issues

1. **Java Version Error**: Ensure Java 11 or higher is installed and in PATH
2. **Gradle Wrapper Missing**: Run `gradle wrapper` to generate the wrapper
3. **Signing Configuration Missing**: Set up signing configuration for Google Play releases
4. **CDN Upload Failed**: Verify Aliyun OSS credentials and bucket permissions
5. **Google Play Upload Failed**: Verify service account permissions and package name

#### Manual Steps

If the script fails, you can run individual steps manually:

```bash
# Build standard debug
./gradlew assembleStandardDebug

# Build Google Play release
./gradlew assembleGoogleplayRelease

# Build Google Play bundle
./gradlew bundleGoogleplayRelease

# Clean build
./gradlew clean
```

### Security Notes

- Never commit `release.config` with actual credentials
- Use environment variables in CI/CD systems
- Keep keystore files secure and backed up
- Rotate access keys regularly

### CI/CD Integration

For CI/CD integration, set the environment variables in your CI/CD system and run:

```bash
source scripts/release.config
./scripts/release.sh
```

The script will automatically detect missing configurations and skip those steps while continuing with available builds. 
import { fileIo } from '@kit.CoreFileKit';
import { image } from '@kit.ImageKit';
import { media } from '@kit.MediaKit';

/**
 * Samples evenly spaced frames from a video and writes them as JPEG files. The LLM prompt then
 * references those frames as <img> tags, which the engine already knows how to embed.
 */
export class VideoFrameExtractor {
  private static instance: VideoFrameExtractor = new VideoFrameExtractor();

  static getInstance(): VideoFrameExtractor {
    return VideoFrameExtractor.instance;
  }

  async extract(videoPath: string, outputDirectory: string, frameCount: number = 8): Promise<string[]> {
    const file = fileIo.openSync(videoPath, fileIo.OpenMode.READ_ONLY);
    const fileSize = fileIo.statSync(videoPath).size;
    const extractor = await media.createAVMetadataExtractor();
    const packer = await image.createImagePacker();
    const paths: string[] = [];
    try {
      extractor.fdSrc = { fd: file.fd, offset: 0, length: fileSize };
      const metadata = await extractor.fetchMetadata();
      let durationMs = Number(metadata.duration || '0');
      if (!(durationMs > 0)) {
        durationMs = frameCount * 500;
      }
      const count = Math.max(1, Math.min(frameCount, 16));
      for (let index = 0; index < count; index++) {
        const timeUs = Math.floor(durationMs * 1000 * (index + 0.5) / count);
        let pixelMap: image.PixelMap | undefined;
        try {
          pixelMap = await extractor.fetchFrameByTime(timeUs, media.AVImageQueryOptions.AV_IMAGE_QUERY_CLOSEST,
            { width: 384, height: 384 });
        } catch (error) {
          break;
        }
        if (!pixelMap) {
          break;
        }
        const buffer = await packer.packing(pixelMap, { format: 'image/jpeg', quality: 85 });
        const path = `${outputDirectory}/frame-${Date.now()}-${index}.jpg`;
        const output = fileIo.openSync(path,
          fileIo.OpenMode.CREATE | fileIo.OpenMode.WRITE_ONLY | fileIo.OpenMode.TRUNC);
        try {
          fileIo.writeSync(output.fd, buffer);
        } finally {
          fileIo.closeSync(output);
        }
        paths.push(path);
        await pixelMap.release();
      }
    } finally {
      await packer.release();
      await extractor.release();
      fileIo.closeSync(file);
    }
    return paths;
  }
}
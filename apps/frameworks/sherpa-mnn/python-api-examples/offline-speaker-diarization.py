#!/usr/bin/env python3
# Copyright (c)  2024  Xiaomi Corporation

"""
This file shows how to use sherpa-onnx Python API for
offline/non-streaming speaker diarization.

Usage:

Step 1: Download a speaker segmentation model

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
for a list of available models. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

Step 2: Download a speaker embedding extractor model

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
for a list of available models. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

Step 3. Download test wave files

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
for a list of available test wave files. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

Step 4. Run it

    python3 ./python-api-examples/offline-speaker-diarization.py

"""
from pathlib import Path

import sherpa_mnn
import soundfile as sf


def init_speaker_diarization(num_speakers: int = -1, cluster_threshold: float = 0.5):
    """
    Args:
      num_speakers:
        If you know the actual number of speakers in the wave file, then please
        specify it. Otherwise, leave it to -1
      cluster_threshold:
        If num_speakers is -1, then this threshold is used for clustering.
        A smaller cluster_threshold leads to more clusters, i.e., more speakers.
        A larger cluster_threshold leads to fewer clusters, i.e., fewer speakers.
    """
    segmentation_model = "./sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    embedding_extractor_model = (
        "./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    )

    config = sherpa_mnn.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_mnn.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_mnn.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation_model
            ),
        ),
        embedding=sherpa_mnn.SpeakerEmbeddingExtractorConfig(
            model=embedding_extractor_model
        ),
        clustering=sherpa_mnn.FastClusteringConfig(
            num_clusters=num_speakers, threshold=cluster_threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError(
            "Please check your config and make sure all required files exist"
        )

    return sherpa_mnn.OfflineSpeakerDiarization(config)


def progress_callback(num_processed_chunk: int, num_total_chunks: int) -> int:
    progress = num_processed_chunk / num_total_chunks * 100
    print(f"Progress: {progress:.3f}%")
    return 0


def main():
    wave_filename = "./0-four-speakers-zh.wav"
    if not Path(wave_filename).is_file():
        raise RuntimeError(f"{wave_filename} does not exist")

    audio, sample_rate = sf.read(wave_filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel

    # Since we know there are 4 speakers in the above test wave file, we use
    # num_speakers 4 here
    sd = init_speaker_diarization(num_speakers=4)
    if sample_rate != sd.sample_rate:
        raise RuntimeError(
            f"Expected samples rate: {sd.sample_rate}, given: {sample_rate}"
        )

    show_progress = True

    if show_progress:
        result = sd.process(audio, callback=progress_callback).sort_by_start_time()
    else:
        result = sd.process(audio).sort_by_start_time()

    for r in result:
        print(f"{r.start:.3f} -- {r.end:.3f} speaker_{r.speaker:02}")
        #  print(r) # this one is simpler


if __name__ == "__main__":
    main()

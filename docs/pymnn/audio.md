# MNN Audio Module

`MNN.audio` provides WAV I/O, spectral features, and window functions using
`MNN.expr.Var` tensors.

## Installation

```bash
pip install MNN
```

The examples below use the public Python package. Its Python wrappers accept
keyword arguments and pass positional arguments to the C extension. Direct
`_mnncengine.audio` calls accept positional arguments only.

This change fixes argument parsing for `spectrogram`, `mel_spectrogram`, and
`melscale_fbanks`. Older packages with the affected bindings may raise an error
for these calls. Use a package built from a revision containing the fixes.

## Quick Start

```python
import MNN.audio as audio

waveform, sample_rate = audio.load("input.wav", sr=16000)
features = audio.fbank(waveform, sample_rate=sample_rate, n_mels=80)
print(features.shape)
```

Run the example from the repository root:

```bash
python pymnn/examples/audio_demo.py
python pymnn/examples/audio_demo.py input.wav --output resampled.wav
```

Without a file, the example runs window generation only. With a file, it also
runs audio I/O, spectrogram, mel, fbank, and Whisper feature extraction.

## Audio I/O

### load

```python
waveform, sample_rate = audio.load(filename, sr=0, frame_offset=0, num_frames=-1)
```

- `filename`: Path to a PCM WAV file. The current reader expects RIFF/WAVE data;
  it is not a general MP3 or FLAC decoder.
- `sr`: Target sample rate; `0` preserves the file's sample rate.
- `frame_offset`: First frame to load.
- `num_frames`: Number of frames; `-1` loads the remaining file.

Returns `(waveform, sample_rate)`. `waveform` is a one-dimensional MNN Var.
For feature extraction, resample to the required rate or pass the returned rate.

### save

```python
success = audio.save("output.wav", waveform, sample_rate)
```

Writes a WAV file and returns a boolean indicating success.

## Window Functions and Constants

```python
hamming = audio.hamming_window(400, periodic=True, alpha=0.54, beta=0.46)
hann = audio.hanning_window(400, periodic=True)
hann_native = audio.hann_window(400, True)
```

`hanning_window` is the Python wrapper for the native `hann_window` function.
The wrapper accepts keywords; the native function is also exported and accepts
positional arguments. The default `periodic` value is false.

| Window constant | Value |
|---|---:|
| `audio.HAMMING` | 0 |
| `audio.HANNING` | 1 |
| `audio.POVEY` | 2 |
| `audio.RECTANGULAR` | 3 |
| `audio.BLACKMAN` | 4 |

These constants are exported by the Python package. The current native
`spectrogram` implementation selects Hamming or Hann for values 0 and 1;
other window values fall back to Hann.

For spectrogram center padding, use `audio.CONSTANT`, `audio.REFLECT`, or
`audio.SYMMETRIC` as appropriate for the padding operation.

## Spectral Features

### spectrogram

```python
spec = audio.spectrogram(
    waveform, n_fft=400, hop_length=160, win_length=400,
    window_type=audio.HANNING, pad_left=0, pad_right=0,
    center=True, normalized=False, pad_mode=audio.REFLECT, power=2.0,
)
```

- `n_fft`: FFT size; defaults to 400.
- `hop_length`: Frame stride; `0` uses `n_fft // 2`.
- `win_length`: Window length; `0` uses `n_fft`.
- `window_type`: A window constant; defaults to `audio.HANNING`.
- `pad_left`, `pad_right`: Constant padding before frame extraction.
- `center`: Whether to pad around the waveform; defaults to false.
- `normalized`: Whether to normalize; defaults to false.
- `pad_mode`: Center padding mode; defaults to `audio.REFLECT`.
- `power`: `2.0` for power, `1.0` for magnitude.

Returns an MNN Var.

### melscale_fbanks

```python
filters = audio.melscale_fbanks(80, 400, 16000, True, False, 0.0, 8000.0)
```

Arguments are `n_mels`, `n_fft`, sample rate, `htk`, `norm`, `f_min`, and `f_max`.
The public wrapper currently spells the sample-rate keyword `sampe_rate`; the
example passes it positionally. Defaults are 16000, true, false, 0.0, and 0.0
respectively. `f_max=0.0` uses half the sample rate.

### mel_spectrogram

```python
mel = audio.mel_spectrogram(
    waveform, 80, 400, sample_rate, htk=True, norm=False,
    f_min=0.0, f_max=0.0, hop_length=160, win_length=400,
    window_type=audio.HANNING, pad_left=0, pad_right=0,
    center=False, normalized=False, pad_mode=audio.REFLECT, power=2.0,
)
```

The first four arguments are waveform, mel-bin count, FFT size, and sample rate.
The wrapper's sample-rate keyword is currently `sampe_rate`, so the example uses
a positional argument. The remaining options are available from Python:

- `htk`: Use the HTK mel scale; defaults to true.
- `norm`: Mel-weight normalization flag; defaults to false. The current native
  implementation uses unit scaling when both `htk` and `norm` are true;
  otherwise it scales each band by `2 / (right_frequency - left_frequency)`.
- `f_min`, `f_max`: Frequency bounds; defaults to 0.0, with `f_max=0.0` meaning
  half the sample rate.
- Spectrogram options have the same defaults as `spectrogram` above.

Returns an MNN Var.

### fbank

```python
features = audio.fbank(
    waveform, sample_rate=16000, n_mels=80, n_fft=400,
    hop_length=160, dither=0.0, preemphasis=0.97,
)
```

Extracts Kaldi-style log-mel features. All optional values shown are defaults.
Use the same sample rate for loading and feature extraction.

### whisper_fbank

```python
features = audio.whisper_fbank(
    waveform, sample_rate=16000, n_mels=128, n_fft=400,
    hop_length=160, chunk_len=30,
)
```

Extracts Whisper-style features. All optional values shown are defaults.
Choose the mel-bin count expected by your model; some Whisper models use 80 bins.

## Tensor Operations

`MNN.numpy` is included in the Python package and supports MNN Var operations.
To use ordinary NumPy, explicitly read the tensor data:

```python
import numpy as np

values = np.asarray(features.read())
print(values.shape)
```

// MNN AUDIO
static PyObject *PyMNNAUDIO_load(PyObject *self, PyObject *args) {
    const char *filename = NULL;
    int sr = 0, frame_offset = 0, num_frames = -1;
    if (PyArg_ParseTuple(args, "s|iii", &filename, &sr, &frame_offset, &num_frames) && filename) {
        return toPyObj<VARP, toPyObj, int, toPyObj>(AUDIO::load(filename, sr, frame_offset, num_frames));
    }
    PyMNN_ERROR("load require args: (string, int, int, int)");
}
static PyObject *PyMNNAUDIO_save(PyObject *self, PyObject *args) {
    const char *filename = NULL;
    PyObject *audio      = nullptr;
    int sample_rate      = 0;
    if (PyArg_ParseTuple(args, "sOi", &filename, &audio, &sample_rate) && filename && isVar(audio)) {
        return toPyObj(AUDIO::save(filename, toVar(audio), sample_rate));
    }
    PyMNN_ERROR("save require args: (string, Var, int)");
}
static PyObject *PyMNNAUDIO_hamming_window(PyObject *self, PyObject *args) {
    int window_size = 0, periodic = 0;
    float alpha = 0.54, beta = 0.46;
    if (PyArg_ParseTuple(args, "i|iff", &window_size, &periodic, &alpha, &beta)) {
        return toPyObj(AUDIO::hamming_window(window_size, periodic, alpha, beta));
    }
    PyMNN_ERROR("hamming_window require args: (int, |bool, float, float)");
}
static PyObject *PyMNNAUDIO_hann_window(PyObject *self, PyObject *args) {
    int window_size = 0, periodic = 0;
    if (PyArg_ParseTuple(args, "i|i", &window_size, &periodic)) {
        return toPyObj(AUDIO::hann_window(window_size, periodic));
    }
    PyMNN_ERROR("hann_window require args: (int, |bool)");
}
static PyObject *PyMNNAUDIO_melscale_fbanks(PyObject *self, PyObject *args) {
    AUDIO::MelscaleParams mel;
    if (PyArg_ParseTuple(args, "ii|ifff", &mel.n_mels, &mel.n_fft, &mel.sample_rate, &mel.htk, &mel.norm, &mel.f_min, &mel.f_max)) {
        return toPyObj(AUDIO::melscale_fbanks(&mel));
    }
    PyMNN_ERROR("melscale_fbanks require args: (int, int, |int, bool, bool, float, float)");
}
static PyObject *PyMNNAUDIO_spectrogram(PyObject *self, PyObject *args) {
    PyObject *waveform = nullptr;
    AUDIO::SpectrogramParams spec;
    if (PyArg_ParseTuple(args, "O|iiiiiiiiiif", &waveform, &spec.n_fft, &spec.hop_length, &spec.win_length,
                         &spec.window_type, &spec.pad_left, &spec.pad_right, &spec.center, &spec.normalized,
                         &spec.pad_mode, &spec.power) &&
        isVar(waveform)) {
        return toPyObj(AUDIO::spectrogram(toVar(waveform), &spec));
    }
    PyMNN_ERROR("spectrogram require args: (Var, |int, int, int, int, int, int, bool, bool, PadValueMode, float)");
}
static PyObject *PyMNNAUDIO_mel_spectrogram(PyObject *self, PyObject *args) {
    PyObject *waveform = nullptr;
    AUDIO::MelscaleParams mel;
    AUDIO::SpectrogramParams spec;
    int n_fft = 400;
    if (PyArg_ParseTuple(args, "O|iiifiiifiiiii", &waveform, &mel.n_mels, &mel.n_fft, &mel.sample_rate, &mel.htk,
                         &mel.norm, &mel.f_min, &mel.f_max, &spec.hop_length, &spec.win_length, &spec.window_type,
                         &spec.pad_left, &spec.pad_right, &spec.center, &spec.normalized, &spec.pad_mode,
                         &spec.power) &&
        isVar(waveform)) {
        spec.n_fft = mel.n_fft;
        return toPyObj(AUDIO::mel_spectrogram(toVar(waveform), &mel, &spec));
    }
    PyMNN_ERROR(
        "mel_spectrogram require args: (Var, |int, bool, bool, float, float, int, int, int, int, int, bool, bool, "
        "PadValueMode, float)"
        "int)");
}
static PyObject *PyMNNAUDIO_fbank(PyObject *self, PyObject *args) {
    PyObject *waveform = nullptr;
    int sample_rate = 16000, n_mels = 80, n_fft = 400, hop_length = 160;
    float dither = 0.0, preemphasis = 0.97;
    if (PyArg_ParseTuple(args, "O|iiiiff", &waveform, &sample_rate, &n_mels, &n_fft, &hop_length, &dither,
                         &preemphasis) &&
        isVar(waveform)) {
        return toPyObj(
            AUDIO::fbank(toVar(waveform), sample_rate, n_mels, n_fft, hop_length, dither, preemphasis));
    }
    PyMNN_ERROR("fbank require args: (Var, |int, int, int, int, float, float)");
}

static PyObject *PyMNNAUDIO_whisper_fbank(PyObject *self, PyObject *args) {
    PyObject *waveform = nullptr;
    int sample_rate = 16000, n_mels = 128, n_fft = 400, hop_length = 160, chunk_len = 30;
    if (PyArg_ParseTuple(args, "O|iiiii", &waveform, &sample_rate, &n_mels, &n_fft, &hop_length, &chunk_len) &&
        isVar(waveform)) {
        return toPyObj(AUDIO::whisper_fbank(toVar(waveform), sample_rate, n_mels, n_fft, hop_length, chunk_len));
    }
    PyMNN_ERROR("whisper_fbank require args: (Var, |int, int, int, int, int)");
}

static PyMethodDef PyMNNAUDIO_methods[] = {
    register_methods(AUDIO,
        load, "load",
        save, "save",
        hamming_window, "hamming_window",
        hann_window, "hann_window",
        melscale_fbanks, "melscale_fbanks",
        spectrogram, "spectrogram",
        mel_spectrogram, "mel_spectrogram",
        fbank, "fbank",
        whisper_fbank, "whisper_fbank"
    )
};

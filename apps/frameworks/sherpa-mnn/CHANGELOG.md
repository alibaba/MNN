## 1.10.46

# Fix kokoro lexicon. (#1886)
# speaker-identification-with-vad-non-streaming-asr.py Lack of support for sense_voice. (#1884)
# Fix generating Chinese lexicon for Kokoro TTS 1.0 (#1888)
# Reduce vad-whisper-c-api example code. (#1891)
# JNI Exception Handling (#1452)
# Fix #1901: UnicodeEncodeError running export_bpe_vocab.py (#1902)
# Fix publishing pre-built windows libraries (#1905)
# Fixing Whisper Model Token Normalization (#1904)
# feat: add mic example for better compatibility (#1909)
# Add onnxruntime 1.18.1 for Linux aarch64 GPU (#1914)
# Add C++ API for streaming zipformer ASR on RK NPU (#1908)
# change [1<<28] to [1<<10], to fix build issues on GOARCH=386 that [1<<28] too large (#1916)
# Flutter Config toJson/fromJson (#1893)
# Fix publishing linux pre-built artifacts (#1919)
# go.mod set to use go 1.17, and use unsafe.Slice to optimize the code (#1920)
# fix: AddPunct panic for Go(#1921)
# Fix publishing macos pre-built artifacts (#1922)
# Minor fixes for rknn (#1925)
# Build wheels for rknn linux aarch64 (#1928)

## 1.10.45

* [update] fixed bug: create golang instance succeed while the c struct create failed (#1860)
* fixed typo in RTF calculations (#1861)
* Export FireRedASR to sherpa-onnx. (#1865)
* Add C++ and Python API for FireRedASR AED models (#1867)
* Add Kotlin and Java API for FireRedAsr AED model (#1870)
* Add C API for FireRedAsr AED model. (#1871)
* Add CXX API for FireRedAsr (#1872)
* Add JavaScript API (node-addon) for FireRedAsr (#1873)
* Add JavaScript API (WebAssembly) for FireRedAsr model. (#1874)
* Add C# API for FireRedAsr Model (#1875)
* Add C# API for FireRedAsr Model (#1875)
* Add Swift API for FireRedAsr AED Model (#1876)
* Add Dart API for FireRedAsr AED Model (#1877)
* Add Go API for FireRedAsr AED Model (#1879)
* Add Pascal API for FireRedAsr AED Model (#1880)

## 1.10.44

* Export MatchaTTS fa-en model to sherpa-onnx (#1832)
* Add C++ support for MatchaTTS models not from icefall. (#1834)
* OfflineRecognizer supports create stream with hotwords (#1833)
* Add PengChengStarling models to sherpa-onnx (#1835)
* Support specifying voice in espeak-ng for kokoro tts models. (#1836)
* Fix: made print sherpa_onnx_loge when it is in debug mode (#1838)
* Add Go API for audio tagging (#1840)
* Fix CI (#1841)
* Update readme to contain links for pre-built Apps (#1853)
* Modify the model used (#1855)
* Flutter OnlinePunctuation (#1854)
* Fix spliting text by languages for kokoro tts. (#1849)

## 1.10.43

* Add MFC example for Kokoro TTS 1.0 (#1815)
* Update sherpa-onnx-tts.js VitsModelConfig.model can be none (#1817)
* Fix passing gb2312 encoded strings to tts on Windows (#1819)
* Support scaling the duration of a pause in TTS. (#1820)
* Fix building wheels for linux aarch64. (#1821)
* Fix CI for Linux aarch64. (#1822)

## 1.10.42

* Fix publishing wheels (#1746)
* Update README to include https://github.com/xinhecuican/QSmartAssistant (#1755)
* Add Kokoro TTS to MFC examples (#1760)
* Refactor node-addon C++ code. (#1768)
* Add keyword spotter C API for HarmonyOS (#1769)
* Add ArkTS API for Keyword spotting. (#1775)
* Add Flutter example for Kokoro TTS (#1776)
* Initialize the audio session for iOS ASR example (#1786)
* Fix: Prepend 0 to tokenization to prevent word skipping for Kokoro. (#1787)
* Export Kokoro 1.0 to sherpa-onnx (#1788)
* Add C++ and Python API for Kokoro 1.0 multilingual TTS model (#1795)
* Add Java and Koltin API for Kokoro TTS 1.0 (#1798)
* Add Android demo for Kokoro TTS 1.0 (#1799)
* Add C API for Kokoro TTS 1.0 (#1801)
* Add CXX API for Kokoro TTS 1.0 (#1802)
* Add Swift API for Kokoro TTS 1.0 (#1803)
* Add Go API for Kokoro TTS 1.0 (#1804)
* Add C# API for Kokoro TTS 1.0 (#1805)
* Add Dart API for Kokoro TTS 1.0 (#1806)
* Add Pascal API for Kokoro TTS 1.0 (#1807)
* Add JavaScript API (node-addon) for Kokoro TTS 1.0 (#1808)
* Add JavaScript API (WebAssembly) for Kokoro TTS 1.0 (#1809)
* Add Flutter example for Kokoro TTS 1.0 (#1810)
* Add iOS demo for Kokoro TTS 1.0 (#1812)
* Add HarmonyOS demo for Kokoro TTS 1.0 (#1813)

## 1.10.41

* Fix UI for Android TTS Engine. (#1735)
* Add iOS TTS example for MatchaTTS (#1736)
* Add iOS example for Kokoro TTS (#1737)
* Fix dither binding in Pybind11 to ensure independence from high_freq in FeatureExtractorConfig (#1739)
* Fix keyword spotting. (#1689)
* Update readme to include https://github.com/hfyydd/sherpa-onnx-server (#1741)
* Reduce vad-moonshine-c-api example code. (#1742)
* Support Kokoro TTS for HarmonyOS. (#1743)

## 1.10.40

* Fix building wheels (#1703)
* Export kokoro to sherpa-onnx (#1713)
* Add C++ and Python API for Kokoro TTS models. (#1715)
* Add C API for Kokoro TTS models (#1717)
* Fix style issues (#1718)
* Add C# API for Kokoro TTS models (#1720)
* Add Swift API for Kokoro TTS models (#1721)
* Add Go API for Kokoro TTS models (#1722)
* Add Dart API for Kokoro TTS models (#1723)
* Add Pascal API for Kokoro TTS models (#1724)
* Add JavaScript API (node-addon) for Kokoro TTS models (#1725)
* Add JavaScript (WebAssembly) API for Kokoro TTS models. (#1726)
* Add Koltin and Java API for Kokoro TTS models (#1728)
* Update README.md for KWS to not use git lfs. (#1729)




## 1.10.39

* Fix building without TTS (#1691)
* Add README for android libs. (#1693)
* Fix: export-onnx.py(expected all tensors to be on the same device) (#1699)
* Fix passing strings from C# to C. (#1701)

## 1.10.38

* Fix initializing TTS in Python. (#1664)
* Remove spaces after punctuations for TTS (#1666)
* Add constructor fromPtr() for all flutter class with factory ctor. (#1667)
* Add Kotlin API for Matcha-TTS models. (#1668)
* Support Matcha-TTS models using espeak-ng (#1672)
* Add Java API for Matcha-TTS models. (#1673)
* Avoid adding tail padding for VAD in generate-subtitles.py (#1674)
* Add C API for MatchaTTS models (#1675)
* Add CXX API for MatchaTTS models (#1676)
* Add JavaScript API (node-addon-api) for MatchaTTS models. (#1677)
* Add HarmonyOS examples for MatchaTTS. (#1678)
* Upgraded to .NET 8 and made code style a little more internally consistent. (#1680)
* Update workflows to use .NET 8.0 also. (#1681)
* Add C# and JavaScript (wasm) API for MatchaTTS models (#1682)
* Add Android demo for MatchaTTS models. (#1683)
* Add Swift API for MatchaTTS models. (#1684)
* Add Go API for MatchaTTS models (#1685)
* Add Pascal API for MatchaTTS models. (#1686)
* Add Dart API for MatchaTTS models (#1687)

## 1.10.37

* Add new tts models for Latvia and Persian+English (#1644)
* Add a byte-level BPE Chinese+English non-streaming zipformer model (#1645)
* Support removing invalid utf-8 sequences. (#1648)
* Add TeleSpeech CTC to non_streaming_server.py (#1649)
* Fix building macOS libs (#1656)
* Add Go API for Keyword spotting (#1662)
* Add Swift online punctuation (#1661)
* Add C++ runtime for Matcha-TTS (#1627)

## 1.10.36

* Update AAR version in Android Java demo (#1618)
* Support linking onnxruntime statically for Android (#1619)
* Update readme to include Open-LLM-VTuber (#1622)
* Rename maxNumStences to maxNumSentences (#1625)
* Support using onnxruntime 1.16.0 with CUDA 11.4 on Jetson Orin NX (Linux arm64 GPU). (#1630)
* Update readme to include jetson orin nx and nano b01 (#1631)
* feat: add checksum action (#1632)
* Support decoding with byte-level BPE (bbpe) models. (#1633)
* feat: enable c api for android ci (#1635)
* Update README.md (#1640)
* SherpaOnnxVadAsr: Offload runSecondPass to background thread for improved real-time audio processing (#1638)
* Fix GitHub actions. (#1642)


## 1.10.35

* Add missing changes about speaker identfication demo for HarmonyOS (#1612)
* Provide sherpa-onnx.aar for Android (#1615)
* Use aar in Android Java demo. (#1616)

## 1.10.34

* Fix building node-addon package (#1598)
* Update doc links for HarmonyOS (#1601)
* Add on-device real-time ASR demo for HarmonyOS (#1606)
* Add speaker identification APIs for HarmonyOS (#1607)
* Add speaker identification demo for HarmonyOS (#1608)
* Add speaker diarization API for HarmonyOS. (#1609)
* Add speaker diarization demo for HarmonyOS (#1610)

## 1.10.33

* Add non-streaming ASR support for HarmonyOS. (#1564)
* Add streaming ASR support for HarmonyOS. (#1565)
* Fix building for Android (#1568)
* Publish `sherpa_onnx.har` for HarmonyOS (#1572)
* Add VAD+ASR demo for HarmonyOS (#1573)
* Fix publishing har packages for HarmonyOS (#1576)
* Add CI to build HAPs for HarmonyOS (#1578)
* Add microphone demo about VAD+ASR for HarmonyOS (#1581)
* Fix getting microphone permission for HarmonyOS VAD+ASR example (#1582)
* Add HarmonyOS support for text-to-speech. (#1584)
* Fix: support both old and new websockets request headers format (#1588)
* Add on-device tex-to-speech (TTS) demo for HarmonyOS (#1590)

## 1.10.32

* Support cross-compiling for HarmonyOS (#1553)
* HarmonyOS support for VAD. (#1561)
* Fix publishing flutter iOS app to appstore (#1563).

## 1.10.31

* Publish pre-built wheels for Python 3.13 (#1485)
* Publish pre-built macos xcframework (#1490)
* Fix reading tokens.txt on Windows. (#1497)
* Add two-pass ASR Android APKs for Moonshine models. (#1499)
* Support building GPU-capable sherpa-onnx on Linux aarch64. (#1500)
* Publish pre-built wheels with CUDA support for Linux aarch64. (#1507)
* Export the English TTS model from MeloTTS (#1509)
* Add Lazarus example for Moonshine models. (#1532)
* Add isolate_tts demo (#1529)
* Add WebAssembly example for VAD + Moonshine models. (#1535)
* Add Android APK for streaming Paraformer ASR (#1538)
* Support static build for windows arm64. (#1539)
* Use xcframework for Flutter iOS plugin to support iOS simulators.

## 1.10.30

* Fix building node-addon for Windows x86. (#1469)
* Begin to support https://github.com/usefulsensors/moonshine (#1470)
* Publish pre-built JNI libs for Linux aarch64 (#1472)
* Add C++ runtime and Python APIs for Moonshine models (#1473)
* Add Kotlin and Java API for Moonshine models (#1474)
* Add C and C++ API for Moonshine models (#1476)
* Add Swift API for Moonshine models. (#1477)
* Add Go API examples for adding punctuations to text. (#1478)
* Add Go API for Moonshine models (#1479)
* Add JavaScript API for Moonshine models (#1480)
* Add Dart API for Moonshine models. (#1481)
* Add Pascal API for Moonshine models (#1482)
* Add C# API for Moonshine models. (#1483)

## 1.10.29

* Add Go API for offline punctuation models (#1434)
* Support https://huggingface.co/Revai/reverb-diarization-v1 (#1437)
* Add more models for speaker diarization (#1440)
* Add Java API example for hotwords. (#1442)
* Add java android demo (#1454)
* Add C++ API for streaming ASR. (#1455)
* Add C++ API for non-streaming ASR (#1456)
* Handle NaN embeddings in speaker diarization. (#1461)
* Add speaker identification with VAD and non-streaming ASR using ALSA (#1463)
* Support GigaAM CTC models for Russian ASR (#1464)
* Add GigaAM NeMo transducer model for Russian ASR (#1467)

## 1.10.28

* Fix swift example for generating subtitles. (#1362)
* Allow more online models to load tokens file from the memory (#1352)
* Fix CI errors introduced by supporting loading keywords from buffers (#1366)
* Fix running MeloTTS models on GPU. (#1379)
* Support Parakeet models from NeMo (#1381)
* Export Pyannote speaker segmentation models to onnx (#1382)
* Support Agglomerative clustering. (#1384)
* Add Python API for clustering (#1385)
* support whisper turbo (#1390)
* context_state is not set correctly when previous context is passed after reset (#1393)
* Speaker diarization example with onnxruntime Python API (#1395)
* C++ API for speaker diarization (#1396)
* Python API for speaker diarization. (#1400)
* C API for speaker diarization (#1402)
* docs(nodejs-addon-examples): add guide for pnpm user (#1401)
* Go API for speaker diarization (#1403)
* Swift API for speaker diarization (#1404)
* Update readme to include more external projects using sherpa-onnx (#1405)
* C# API for speaker diarization (#1407)
* JavaScript API (node-addon) for speaker diarization (#1408)
* WebAssembly exmaple for speaker diarization (#1411)
* Handle audio files less than 10s long for speaker diarization. (#1412)
* JavaScript API with WebAssembly for speaker diarization (#1414)
* Kotlin API for speaker diarization (#1415)
* Java API for speaker diarization (#1416)
* Dart API for speaker diarization (#1418)
* Pascal API for speaker diarization (#1420)
* Android JNI support for speaker diarization (#1421)
* Android demo for speaker diarization (#1423)

## 1.10.27

* Add non-streaming ONNX models for Russian ASR (#1358)
* Fix building Flutter TTS examples for Linux (#1356)
* Support passing utf-8 strings from JavaScript to C++. (#1355)
* Fix sherpa_onnx.go to support returning empty recognition results (#1353)

## 1.10.26

* Add links to projects using sherpa-onnx. (#1345)
* Support lang/emotion/event results from SenseVoice in Swift API. (#1346)
* Support specifying max speech duration for VAD. (#1348)
* Add APIs about max speech duration in VAD for various programming languages (#1349)

## 1.10.25

* Allow tokens and hotwords to be loaded from buffered string driectly (#1339)
* Fix computing features for CED audio tagging models. (#1341)
* Preserve previous result as context for next segment (#1335)
* Add Python binding for online punctuation models (#1312)
* Fix vad.Flush(). (#1329)
* Fix wasm app for streaming paraformer (#1328)
* Build websocket related binaries for embedded systems. (#1327)
* Fixed the C api calls and created the TTS project file (#1324)
* Re-implement LM rescore for online transducer (#1231)

## 1.10.24

* Add VAD and keyword spotting for the Node package with WebAssembly (#1286)
* Fix releasing npm package and fix building Android VAD+ASR example (#1288)
* add Tokens []string, Timestamps []float32, Lang string, Emotion string, Event string (#1277)
* add vad+sense voice example for C API (#1291)
* ADD VAD+ASR example for dart with CircularBuffer. (#1293)
* Fix VAD+ASR example for Dart API. (#1294)
* Avoid SherpaOnnxSpeakerEmbeddingManagerFreeBestMatches freeing null. (#1296)
* Fix releasing wasm app for vad+asr (#1300)
* remove extra files from linux/macos/windows jni libs (#1301)
* two-pass Android APK for SenseVoice (#1302)
* Downgrade flutter sdk versions. (#1305)
* Reduce onnxruntime log output. (#1306)
* Provide prebuilt .jar files for different java versions. (#1307)


## 1.10.23

* flutter: add lang, emotion, event to OfflineRecognizerResult (#1268)
* Use a separate thread to initialize models for lazarus examples. (#1270)
* Object pascal examples for recording and playing audio with portaudio. (#1271)
* Text to speech API for Object Pascal. (#1273)
* update kotlin api for better release native object and add user-friendly apis. (#1275)
* Update wave-reader.cc to support 8/16/32-bit waves (#1278)
* Add WebAssembly for VAD (#1281)
* WebAssembly example for VAD + Non-streaming ASR (#1284)

## 1.10.22

* Add Pascal API for reading wave files (#1243)
* Pascal API for streaming ASR (#1246)
* Pascal API for non-streaming ASR (#1247)
* Pascal API for VAD (#1249)
* Add more C API examples (#1255)
* Add emotion, event of SenseVoice. (#1257)
* Support reading multi-channel wave files with 8/16/32-bit encoded samples (#1258)
* Enable IPO only for Release build. (#1261)
* Add Lazarus example for generating subtitles using Silero VAD with non-streaming ASR (#1251)
* Fix looking up OOVs in lexicon.txt for MeloTTS models. (#1266)


## 1.10.21

* Fix ffmpeg c api example (#1185)
* Fix splitting sentences for MeloTTS (#1186)
* Non-streaming WebSocket client for Java. (#1190)
* Fix copying asset files for flutter examples. (#1191)
* Add Chinese+English tts example for flutter (#1192)
* Add speaker identification and verification exmaple for Dart API (#1194)
* Fix reading non-standard wav files. (#1199)
* Add ReazonSpeech Japanese pre-trained model (#1203)
* Describe how to add new words for MeloTTS models (#1209)
* Remove libonnxruntime_providers_cuda.so as a dependency. (#1210)
* Fix setting SenseVoice language. (#1214)
* Support passing TTS callback in Swift API (#1218)
* Add MeloTTS example for ios (#1223)
* Add online punctuation and casing prediction model for English language (#1224)
* Fix python two pass ASR examples (#1230)
* Add blank penalty for various language bindings

## 1.10.20

* Add Dart API for audio tagging
* Add Dart API for adding punctuations to text

## 1.10.19

* Prefix all C API functions with SherpaOnnx

## 1.10.18

* Fix the case when recognition results contain the symbol `"`. It caused
  issues when converting results to a json string.

## 1.10.17

* Support SenseVoice CTC models.
* Add Dart API for keyword spotter.

## 1.10.16

* Support zh-en TTS model from MeloTTS.

## 1.10.15

* Downgrade onnxruntime from v1.18.1 to v1.17.1

## 1.10.14

* Support whisper large v3
* Update onnxruntime from v1.18.0 to v1.18.1
* Fix invalid utf8 sequence from Whisper for Dart API.

## 1.10.13

* Update onnxruntime from 1.17.1 to 1.18.0
* Add C# API for Keyword spotting

## 1.10.12

* Add Flush to VAD so that the last speech segment can be detected. See also
  https://github.com/k2-fsa/sherpa-onnx/discussions/1077#discussioncomment-9979740

## 1.10.11

* Support the iOS platform for Flutter.

## 1.10.10

* Build sherpa-onnx into a single shared library.

## 1.10.9

* Fix released packages. piper-phonemize was not included in v1.10.8.

## 1.10.8

* Fix released packages. There should be a lib directory.

## 1.10.7

* Support Android for Flutter.

## 1.10.2

* Fix passing C# string to C++

## 1.10.1

* Enable to stop TTS generation

## 1.10.0

* Add inverse text normalization

## 1.9.30

* Add TTS

## 1.9.29

* Publish with CI

## 0.0.3

* Fix path separator on Windows.

## 0.0.2

* Support specifying lib path.

## 0.0.1

* Initial release.

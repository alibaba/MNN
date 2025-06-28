# File descriptions

- [./sherpa-onnx-alsa.cc](./sherpa-onnx-alsa.cc) For Linux only, especially for
  embedded Linux, e.g., Raspberry Pi; it uses a streaming model for real-time
  speech recognition with a microphone.

- [./sherpa-onnx-microphone.cc](./sherpa-onnx-microphone.cc)
  For Linux/Windows/macOS; it uses a streaming model for real-time speech
  recognition with a microphone.

- [./sherpa-onnx-microphone-offline.cc](./sherpa-onnx-microphone-offline.cc)
  For Linux/Windows/macOS; it uses a non-streaming model for speech
  recognition with a microphone.

- [./sherpa-onnx.cc](./sherpa-onnx.cc)
  It uses a streaming model to decode wave files

- [./sherpa-onnx-offline.cc](./sherpa-onnx-offline.cc)
  It uses a non-streaming model to decode wave files

- [./online-websocket-server.cc](./online-websocket-server.cc)
  WebSocket server for streaming models.

- [./offline-websocket-server.cc](./offline-websocket-server.cc)
  WebSocket server for non-streaming models.

- [./sherpa-onnx-vad-microphone.cc](./sherpa-onnx-vad-microphone.cc)
  Use silero VAD to detect speeches with a microphone.


# Sana Cartoon-Style Image Edit (MNN)

Sana is integrated into MNN for cartoon-style photo editing and face-style conversion on mobile/edge.

[中文文档](./README_ZH.md)

## Showcase

<p align="center">
  <img width="40%" alt="Sana Showcase" src="https://meta.alicdn.com/data/mnn/assets/sana_showcase_android.jpg" style="margin: 0 10px;">
  <img width="40%" alt="Sana Home Screenshot" src="https://meta.alicdn.com/data/mnn/assets/sana_showcase_ios.jpg" style="margin: 0 10px;">
</p>

## App Links
- [Android MNN LLM Chat](../../apps/Android/MnnLlmChat/README.md)
- [iOS MNN LLM Chat](../../apps/iOS/MNNLLMChat/README.md)

## Model Links
Cartoon-style edit model

- HuggingFace: [https://huggingface.co/taobao-mnn/MNN-Sana-Edit-V2](https://huggingface.co/taobao-mnn/MNN-Sana-Edit-V2)
- ModelScope: [https://modelscope.cn/models/MNN/MNN-Sana-Edit-V2](https://modelscope.cn/models/MNN/MNN-Sana-Edit-V2)

## Recommended Settings
- Input: use square images when possible; non-square input often gives worse results.
- Output: fixed at `512x512`.
- Prompt: fixed for this model pipeline. No extra prompt is needed; changing it may reduce quality.
- Step: `10` steps are recommended. Fewer steps usually reduce visual quality.

## Usage in MNN Chat App
1. Open `apps/Android/MnnLlmChat`.
2. Download model `MNN-Sana-Edit-V2` from model market.
3. Switch to Sana model in app.
4. Upload an input face image (square is preferred).
5. Keep prompt/settings as default and set step to `10`.
6. Run generation and review `512x512` output.


## Reference
- Sana paper: [https://arxiv.org/abs/2410.10629](https://arxiv.org/abs/2410.10629)
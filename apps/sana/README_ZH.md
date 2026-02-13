# Sana 卡通风格图像编辑 (MNN)

Sana 已集成到 MNN，用于移动端/边缘端的卡通风格图像编辑与人脸风格转换。

English: [README.md](./README.md)

## 效果展示

<p align="center">
  <img width="40%" alt="Sana 效果展示" src="https://meta.alicdn.com/data/mnn/assets/sana_showcase_android.jpg" style="margin: 0 10px;">
  <img width="40%" alt="Sana 首页截图" src="https://meta.alicdn.com/data/mnn/assets/sana_showcase_ios.jpg" style="margin: 0 10px;">
</p>

## 应用链接
- [Android MNN LLM Chat](../../apps/Android/MnnLlmChat/README.md)
- [iOS MNN LLM Chat](../../apps/iOS/MNNLLMChat/README.md)

## 模型链接
卡通风格编辑模型。

- HuggingFace: [https://huggingface.co/taobao-mnn/MNN-Sana-Edit-V2](https://huggingface.co/taobao-mnn/MNN-Sana-Edit-V2)
- ModelScope: [https://modelscope.cn/models/MNN/MNN-Sana-Edit-V2](https://modelscope.cn/models/MNN/MNN-Sana-Edit-V2)

## 推荐设置
- 输入：尽量使用正方形图片；非正方形输入往往效果较差。
- 输出：固定为 `512x512`。
- 提示词：本模型流程中已固定，无需额外填写；修改可能降低效果。
- 步数：建议使用 `10` 步。步数过少通常会影响画质。

## 在 MNN Chat 应用中的使用
1. 打开 `apps/Android/MnnLlmChat`。
2. 从模型市场下载 `MNN-Sana-Edit-V2` 模型。
3. 在应用中切换到 Sana 模型。
4. 上传一张人脸输入图（建议为正方形）。
5. 保持提示词与设置默认，将步数设为 `10`。
6. 运行生成并查看 `512x512` 输出结果。

## 参考
- Sana 论文: [https://arxiv.org/abs/2410.10629](https://arxiv.org/abs/2410.10629)

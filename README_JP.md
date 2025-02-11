![MNN](doc/banner.png)

[English Version](README.md)

[中文版本](README_CN.md)

[MNN ホームページ](http://www.mnn.zone)

## ニュース 🔥
- [2025/01/23] フルマルチモーダル LLM Android アプリをリリースしました: [MNN-LLM-Android](./project/android/apps/MnnLlmApp/README.md)。テキストからテキスト、画像からテキスト、音声からテキスト、テキストから画像生成を含みます。
<p align="center">
  <img width="20%" alt="Icon"  src="./project/android/apps/MnnLlmApp/assets/image_home.jpg" style="margin: 0 10px;">
  <img width="20%" alt="Icon" src="./project/android/apps/MnnLlmApp/assets/image_diffusion.jpg" style="margin: 0 10px;">
  <img width="20%" alt="Icon" src="./project/android/apps/MnnLlmApp/assets/image_sound.jpg" style="margin: 0 10px;">
  <img width="20%" alt="Icon" src="./project/android/apps/MnnLlmApp/assets/image_image.jpg" style="margin: 0 10px;">
</p>

## 紹介
MNNは非常に効率的で軽量なディープラーニングフレームワークです。ディープラーニングモデルの推論とトレーニングをサポートし、デバイス上での推論とトレーニングにおいて業界をリードするパフォーマンスを発揮します。現在、MNNはAlibaba Incの30以上のアプリに統合されており、ライブ放送、短編動画キャプチャ、検索推薦、画像による商品検索、インタラクティブマーケティング、エクイティ配布、セキュリティリスク管理など、70以上の使用シナリオをカバーしています。さらに、MNNはIoTなどの組み込みデバイスでも使用されています。

[MNN-LLM](https://github.com/alibaba/MNN/tree/master/transformers/llm)は、MNNエンジンをベースに開発された大規模言語モデルのランタイムソリューションです。このプロジェクトの使命は、LLMモデルをすべてのプラットフォーム（モバイルフォン/PC/IOT）にローカルにデプロイすることです。Qianwen、Baichuan、Zhipu、LLAMAなどの人気のある大規模言語モデルをサポートしています。[MNN-LLMユーザーガイド](https://mnn-docs.readthedocs.io/en/latest/transformers/llm.html)

[MNN-Diffusion](https://github.com/alibaba/MNN/tree/master/transformers/diffusion)は、MNNエンジンをベースに開発されたStable Diffusionモデルのランタイムソリューションです。このプロジェクトの使命は、Stable Diffusionモデルをすべてのプラットフォームにローカルにデプロイすることです。[MNN-Diffusionユーザーガイド](https://mnn-docs.readthedocs.io/en/latest/transformers/diffusion.html)

![アーキテクチャ](doc/architecture.png)

Alibabaの内部では、[MNN](https://mp.weixin.qq.com/s/5I1ISpx8lQqvCS8tGd6EJw)は、[Walle](https://mp.weixin.qq.com/s/qpeCETty0BqqNJV9CMJafA)システムの基本モジュールとして機能しています。Walleは、エンドツーエンド、汎用、大規模なデバイスクラウド協調機械学習のための最初の生産システムであり、トップシステム会議OSDI'22で発表されました。MNNの主要な設計原則と、広範なベンチマークテスト結果（TensorFlow、TensorFlow Lite、PyTorch、PyTorch Mobile、TVMとの比較）は、OSDI論文に記載されています。ベンチマークテストのスクリプトと手順は、「/benchmark」パスに配置されています。MNNまたはWalleの設計が研究や生産に役立つ場合は、以下のようにOSDI論文を引用してください。

    @inproceedings {proc:osdi22:walle,
        author = {Chengfei Lv and Chaoyue Niu and Renjie Gu and Xiaotang Jiang and Zhaode Wang and Bin Liu and Ziqi Wu and Qiulin Yao and Congyu Huang and Panos Huang and Tao Huang and Hui Shu and Jinde Song and Bin Zou and Peng Lan and Guohuan Xu and Fei Wu and Shaojie Tang and Fan Wu and Guihai Chen},
        title = {Walle: An {End-to-End}, {General-Purpose}, and {Large-Scale} Production System for {Device-Cloud} Collaborative Machine Learning},
        booktitle = {16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)},
        year = {2022},
        isbn = {978-1-939133-28-1},
        address = {Carlsbad, CA},
        pages = {249--265},
        url = {https://www.usenix.org/conference/osdi22/presentation/lv},
        publisher = {USENIX Association},
        month = jul,
    }


## ドキュメントとワークベンチ
MNNのドキュメントは[Read the docs](https://mnn-docs.readthedocs.io/en/latest)にあります。

docs/READMEを読んで、ドキュメントのHTMLをビルドすることもできます。

MNN Workbenchは[MNNのホームページ](http://www.mnn.zone)からダウンロードできます。事前トレーニング済みのモデル、可視化されたトレーニングツール、デバイスへのワンクリックデプロイメントを提供します。

## 主な特徴
### 軽量
- デバイス向けに最適化されており、依存関係がなく、モバイルデバイスやさまざまな組み込みデバイスに簡単にデプロイできます。
- iOSプラットフォーム：armv7+arm64プラットフォーム用のフルオプションの静的ライブラリサイズは約12MB、リンクされた実行ファイルのサイズ増加は約2Mです。
- Androidプラットフォーム：コアsoサイズは約800KB（armv7a - c++_shared）。
- MNN_BUILD_MINIを使用すると、パッケージサイズを約25％削減できます。入力サイズが固定されている場合に限ります。
- FP16 / Int8量子化をサポートし、モデルサイズを50％〜70％削減できます。

### 汎用性
- `Tensorflow`、`Caffe`、`ONNX`、`Torchscripts`をサポートし、`CNN`、`RNN`、`GAN`、`Transformer`などの一般的なニューラルネットワークをサポートします。
- 複数の入力または出力を持つAIモデル、あらゆる種類の次元形式、動的入力、制御フローをサポートします。
- MNNは、AIモデルで使用されるほぼすべてのOPをサポートしています。コンバータは178の`Tensorflow` OP、52の`Caffe` OP、163の`Torchscripts` OP、158の`ONNX` OPをサポートしています。
- iOS 8.0+、Android 4.3+、およびPOSIXインターフェースを持つ組み込みデバイスをサポートします。
- 複数のデバイスでのハイブリッドコンピューティングをサポートします。現在、CPUとGPUをサポートしています。

### 高性能
- ARM / x64 CPUの性能を最大限に引き出すために、多くの最適化されたアセンブリコードを実装しています。
- Metal / OpenCL / Vulkanを使用して、モバイルデバイス上でのGPU推論をサポートします。
- CUDAとtensorcoreを使用して、NVIDIA GPUを使用したより高速な推論をサポートします。
- 畳み込みおよび転置畳み込みアルゴリズムは効率的で安定しています。Winograd畳み込みアルゴリズムは、3x3、4x4、5x5、6x6、7x7などの対称畳み込みをより良くするために広く使用されています。
- FP16の半精度計算サポートを備えた新しいアーキテクチャARM v8.2を使用すると、速度が2倍になります。ARM v8.2およびVNNI用のsdotを使用すると、2.5倍の速度向上が得られます。

### 使いやすさ
- MNNのOPを使用して、numpyのような数値計算を行うことができます。
- OpenCVのような軽量の画像処理モジュールをサポートし、サイズはわずか100kです。
- PC /モバイルでモデルを構築してトレーニングすることをサポートします。
- MNN Python APIを使用すると、MLエンジ��アはC++コードに触れることなく、MNNを使用して推論、トレーニング、画像処理を簡単に行うことができます。

MNNがサポートするアーキテクチャ/精度は以下の通りです：

- S ：サポートされており、深く最適化されており、使用を推奨します。
- A ：サポートされており、初期の最適化が行われているか、使用可能です。
- B ：サポートされていますが、バグがあるか、最適化されていないため、使用を推奨しません。
- C ：サポートされていません。

| アーキテクチャ / 精度 |  | 通常 | FP16 | BF16 | Int8 |
| --- | --- | --- | --- | --- | --- |
| CPU | ネイティブ | B | C | B | B |
|  | x86/x64-SSE4.1 | A | B | B | A |
|  | x86/x64-AVX2 | S | B | B | A |
|  | x86/x64-AVX512 | S | B | B | S |
|  | ARMv7a | S | S (ARMv8.2) | S | S |
|  | ARMv8 | S | S (ARMv8.2) | S(ARMv8.6) | S |
| GPU | OpenCL | A | S | C | C |
|  | Vulkan | A | A | C | C |
|  | Metal | A | S | C | C |
|  | CUDA | A | S | C | C |
| NPU | CoreML | B | B | C | C |
|  | HIAI | B | C | C | B |
|  | NNAPI | B | B | C | C |



## ツール

MNN（テンソル計算エンジン）に基づいて、推論、トレーニング、一般的な計算をサポートする一連のツールを提供しています。

- MNN-Converter：他のモデルをMNNモデルに変換して推論を行うためのツール。Tensorflow���lite）、Caffe、ONNX、Torchscriptsなどをサポートし、グラフの最適化を行って計算を削減します。
- MNN-Compress：モデルを圧縮してサイズを削減し、パフォーマンス/速度を向上させます。
- MNN-Express：制御フローを持つモデルをサポートし、MNNのOPを使用して一般的な計算を行います。
- MNN-CV：OpenCVに似たライブラリですが、MNNに基づいており、はるかに軽量です。
- MNN-Train：MNNモデルのトレーニングをサポートします。

## MNNコミュニティからのディスカッションとヘルプの取得方法

グループディスカッションは主に中国語で行われますが、英語を話す方も歓迎し、サポートします。

Dingtalkディスカッショングループ：

グループ＃1（満員）：23329087

グループ＃2（満員）：23350225

グループ＃3：QRコード：

![MNN-3](doc/dingdingmnn3.png)

## 歴史的な論文

MNNの初期バージョンは、モバイル推論エンジンとして、手動最適化に焦点を当てたもので、MLSys 2020で発表されました。以前にMNNが研究に役立った場合は、以下のように論文を引用してください：

    @inproceedings{alibaba2020mnn,
      author = {Jiang, Xiaotang and Wang, Huan and Chen, Yiliu and Wu, Ziqi and Wang, Lichuan and Zou, Bin and Yang, Yafeng and Cui, Zongyang and Cai, Yu and Yu, Tianhang and Lv, Chengfei and Wu, Zhihua},
      title = {MNN: A Universal and Efficient Inference Engine},
      booktitle = {MLSys},
      year = {2020}
    }


## ライセンス
Apache 2.0

## 謝辞
MNNの参加者：淘宝技術部、検索エンジニアリングチーム、DAMOチーム、優酷およびその他のAlibaba Groupの従業員。

MNNは以下のプロジェクトを参照しています：
- [Caffe](https://github.com/BVLC/caffe)
- [flatbuffer](https://github.com/google/flatbuffers)
- [gemmlowp](https://github.com/google/gemmlowp)
- [Google Vulkan demo](http://www.github.com/googlesamples/android-vulkan-tutorials)
- [Halide](https://github.com/halide/Halide)
- [Mace](https://github.com/XiaoMi/mace)
- [ONNX](https://github.com/onnx/onnx)
- [protobuffer](https://github.com/protocolbuffers/protobuf)
- [skia](https://github.com/google/skia)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [ncnn](https://github.com/Tencent/ncnn)
- [paddle-mobile](https://github.com/PaddlePaddle/paddle-mobile)
- [stb](https://github.com/nothings/stb)
- [rapidjson](https://github.com/Tencent/rapidjson)
- [pybind11](https://github.com/pybind/pybind11)
- [pytorch](https://github.com/pytorch/pytorch)
- [bolt](https://github.com/huawei-noah/bolt)
- [libyuv](https://chromium.googlesource.com/libyuv/libyuv)
- [libjpeg](https://github.com/libjpeg-turbo/libjpeg-turbo)
- [opencv](https://github.com/opencv/opencv)

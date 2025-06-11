#!/usr/bin/env python3

"""
This script shows how to add punctuations to text using sherpa-onnx Python API.

Please download the model from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models

The following is an example

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
"""

from pathlib import Path

import sherpa_mnn


def main():
    model = "./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx"
    if not Path(model).is_file():
        raise ValueError(f"{model} does not exist")
    config = sherpa_mnn.OfflinePunctuationConfig(
        model=sherpa_mnn.OfflinePunctuationModelConfig(ct_transformer=model),
    )

    punct = sherpa_mnn.OfflinePunctuation(config)

    text_list = [
        "这是一个测试你好吗How are you我很好thank you are you ok谢谢你",
        "我们都是木头人不会说话不会动",
        "The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry",
    ]
    for text in text_list:
        text_with_punct = punct.add_punctuation(text)
        print("----------")
        print(f"input: {text}")
        print(f"output: {text_with_punct}")

    print("----------")


if __name__ == "__main__":
    main()

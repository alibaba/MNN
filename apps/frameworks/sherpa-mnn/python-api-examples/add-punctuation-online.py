#!/usr/bin/env python3

"""
This script shows how to add punctuations to text using sherpa-onnx Python API.

Please download the model from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models

The following is an example

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
tar xvf sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
rm sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
"""

from pathlib import Path

import sherpa_mnn


def main():
    model = "./sherpa-onnx-online-punct-en-2024-08-06/model.onnx"
    bpe = "./sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab"
    if not Path(model).is_file():
        raise ValueError(f"{model} does not exist")
    if not Path(bpe).is_file():
        raise ValueError(f"{bpe} does not exist")

    model_config = sherpa_mnn.OnlinePunctuationModelConfig(
        cnn_bilstm=model, bpe_vocab=bpe
    )
    config = sherpa_mnn.OnlinePunctuationConfig(model_config=model_config)
    punct = sherpa_mnn.OnlinePunctuation(config)

    texts = [
        "how are you i am fine thank you",
        "The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry",
    ]
    for text in texts:
        text_with_punct = punct.add_punctuation_with_case(text)
        print("----------")
        print(f"input : {text}")
        print(f"output: {text_with_punct}")
    print("----------")


if __name__ == "__main__":
    main()

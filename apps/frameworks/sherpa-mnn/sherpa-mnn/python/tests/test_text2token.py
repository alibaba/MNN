# sherpa-mnn/python/tests/test_text2token.py
#
# Copyright (c)  2023  Xiaomi Corporation
#
# To run this single test, use
#
#  ctest --verbose -R  test_text2token_py

import unittest
from pathlib import Path

import sherpa_mnn

d = "/tmp/sherpa-test-data"
# Please refer to
# https://github.com/pkufool/sherpa-test-data
# to download test data for testing


class TestText2Token(unittest.TestCase):
    def test_bpe(self):
        tokens = f"{d}/text2token/tokens_en.txt"
        bpe_model = f"{d}/text2token/bpe_en.model"

        if not Path(tokens).is_file() or not Path(bpe_model).is_file():
            print(
                f"No test data found, skipping test_bpe().\n"
                f"You can download the test data by: \n"
                f"git clone https://github.com/pkufool/sherpa-test-data.git /tmp/sherpa-test-data"
            )
            return

        texts = ["HELLO WORLD", "I LOVE YOU"]
        encoded_texts = sherpa_mnn.text2token(
            texts,
            tokens=tokens,
            tokens_type="bpe",
            bpe_model=bpe_model,
        )
        assert encoded_texts == [
            ["▁HE", "LL", "O", "▁WORLD"],
            ["▁I", "▁LOVE", "▁YOU"],
        ], encoded_texts

        encoded_ids = sherpa_mnn.text2token(
            texts,
            tokens=tokens,
            tokens_type="bpe",
            bpe_model=bpe_model,
            output_ids=True,
        )
        assert encoded_ids == [[22, 58, 24, 425], [19, 370, 47]], encoded_ids

    def test_cjkchar(self):
        tokens = f"{d}/text2token/tokens_cn.txt"

        if not Path(tokens).is_file():
            print(
                f"No test data found, skipping test_cjkchar().\n"
                f"You can download the test data by: \n"
                f"git clone https://github.com/pkufool/sherpa-test-data.git /tmp/sherpa-test-data"
            )
            return

        texts = ["世界人民大团结", "中国 VS 美国"]
        encoded_texts = sherpa_mnn.text2token(
            texts, tokens=tokens, tokens_type="cjkchar"
        )
        assert encoded_texts == [
            ["世", "界", "人", "民", "大", "团", "结"],
            ["中", "国", "V", "S", "美", "国"],
        ], encoded_texts
        encoded_ids = sherpa_mnn.text2token(
            texts,
            tokens=tokens,
            tokens_type="cjkchar",
            output_ids=True,
        )
        assert encoded_ids == [
            [379, 380, 72, 874, 93, 1251, 489],
            [262, 147, 3423, 2476, 21, 147],
        ], encoded_ids

    def test_cjkchar_bpe(self):
        tokens = f"{d}/text2token/tokens_mix.txt"
        bpe_model = f"{d}/text2token/bpe_mix.model"

        if not Path(tokens).is_file() or not Path(bpe_model).is_file():
            print(
                f"No test data found, skipping test_cjkchar_bpe().\n"
                f"You can download the test data by: \n"
                f"git clone https://github.com/pkufool/sherpa-test-data.git /tmp/sherpa-test-data"
            )
            return

        texts = ["世界人民 GOES TOGETHER", "中国 GOES WITH 美国"]
        encoded_texts = sherpa_mnn.text2token(
            texts,
            tokens=tokens,
            tokens_type="cjkchar+bpe",
            bpe_model=bpe_model,
        )
        assert encoded_texts == [
            ["世", "界", "人", "民", "▁GO", "ES", "▁TOGETHER"],
            ["中", "国", "▁GO", "ES", "▁WITH", "美", "国"],
        ], encoded_texts
        encoded_ids = sherpa_mnn.text2token(
            texts,
            tokens=tokens,
            tokens_type="cjkchar+bpe",
            bpe_model=bpe_model,
            output_ids=True,
        )
        assert encoded_ids == [
            [1368, 1392, 557, 680, 275, 178, 475],
            [685, 736, 275, 178, 179, 921, 736],
        ], encoded_ids


if __name__ == "__main__":
    unittest.main()

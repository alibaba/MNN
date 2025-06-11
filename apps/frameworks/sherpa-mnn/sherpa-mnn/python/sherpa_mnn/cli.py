# Copyright (c)  2023  Xiaomi Corporation

import logging
try:
    import click
except ImportError:
    print('Please run')
    print('  pip install click')
    print('before you continue')
    raise

from pathlib import Path
from sherpa_mnn import text2token


@click.group()
def cli():
    """
    The shell entry point to sherpa-mnn.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )


@cli.command(name="text2token")
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path())
@click.option(
    "--tokens",
    type=str,
    required=True,
    help="The path to tokens.txt.",
)
@click.option(
    "--tokens-type",
    type=click.Choice(
        ["cjkchar", "bpe", "cjkchar+bpe", "fpinyin", "ppinyin"], case_sensitive=True
    ),
    required=True,
    help="""The type of modeling units, should be cjkchar, bpe, cjkchar+bpe, fpinyin or ppinyin.
    fpinyin means full pinyin, each cjkchar has a pinyin(with tone).
    ppinyin means partial pinyin, it splits pinyin into initial and final,
    """,
)
@click.option(
    "--bpe-model",
    type=str,
    help="The path to bpe.model. Only required when tokens-type is bpe or cjkchar+bpe.",
)
def encode_text(
    input: Path, output: Path, tokens: Path, tokens_type: str, bpe_model: Path
):
    """
    Encode the texts given by the INPUT to tokens and write the results to the OUTPUT.
    Each line in the texts contains the original phrase, it might also contain some
    extra items, for example, the boosting score (startting with :), the triggering
    threshold (startting with #, only used in keyword spotting task) and the original
    phrase (startting with @). Note: the extra items will be kept same in the output.

    example input 1 (tokens_type = ppinyin):

    小爱同学 :2.0 #0.6 @小爱同学
    你好问问 :3.5 @你好问问
    小艺小艺 #0.6 @小艺小艺

    example output 1:

    x iǎo ài t óng x ué :2.0 #0.6 @小爱同学
    n ǐ h ǎo w èn w èn :3.5 @你好问问
    x iǎo y ì x iǎo y ì #0.6 @小艺小艺

    example input 2 (tokens_type = bpe):

    HELLO WORLD :1.5 #0.4
    HI GOOGLE :2.0 #0.8
    HEY SIRI #0.35

    example output 2:

    ▁HE LL O ▁WORLD :1.5 #0.4
    ▁HI ▁GO O G LE :2.0 #0.8
    ▁HE Y ▁S I RI #0.35
    """
    texts = []
    # extra information like boosting score (start with :), triggering threshold (start with #)
    # original keyword (start with @)
    extra_info = []
    with open(input, "r", encoding="utf8") as f:
        for line in f:
            extra = []
            text = []
            toks = line.strip().split()
            for tok in toks:
                if tok[0] == ":" or tok[0] == "#" or tok[0] == "@":
                    extra.append(tok)
                else:
                    text.append(tok)
            texts.append(" ".join(text))
            extra_info.append(extra)

    encoded_texts = text2token(
        texts, tokens=tokens, tokens_type=tokens_type, bpe_model=bpe_model
    )
    with open(output, "w", encoding="utf8") as f:
        for i, txt in enumerate(encoded_texts):
            txt += extra_info[i]
            f.write(" ".join(txt) + "\n")

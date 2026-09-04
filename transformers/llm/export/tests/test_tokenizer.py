from utils.tokenizer import AutoTokenizer, LlmTokenizer, PreTrainedTokenizer


class SplitUnicodeTokenizer:
    eos_token_id = None
    generation_config = None
    chat_template = None
    vocab_size = 2

    def encode(self, text, **kwargs):
        return []

    def decode(self, token_id):
        return "\ufffd"

    def convert_ids_to_tokens(self, token_ids):
        return token_ids

    def convert_tokens_to_string(self, tokens):
        return "\u4f60" if tokens == [1, 2] else "\ufffd"

    def get_vocab(self):
        return {}


def test_id_to_str_buffers_split_unicode_tokens(monkeypatch, tmp_path):
    monkeypatch.setattr(PreTrainedTokenizer, "__init__", lambda self, **kwargs: None)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: SplitUnicodeTokenizer(),
    )
    tokenizer = LlmTokenizer(tmp_path, "test")

    assert tokenizer.id_to_str(1) == ""
    assert tokenizer.id_to_str(2) == "\u4f60"
    assert tokenizer.decode_buffer == []

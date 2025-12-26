import os
import base64
from transformers import PreTrainedTokenizer, AutoTokenizer

class LlmTokenizer(PreTrainedTokenizer):
    def __init__(self, tokenizer_path, model_type, **kwargs):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
        self.tokenizer_path = tokenizer_path
        self.model_type = model_type
        # stop_ids
        self.stop_ids = []
        self.stop_ids.append(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, 'im_end_id'):
            self.stop_ids.append(self.tokenizer.im_end_id)
        try:
            eot_id = self.tokenizer.encode('<|eot_id|>')
            if len(eot_id) == 1:
                self.stop_ids.append(eot_id[0])
            eot_id = self.tokenizer.encode('<end_of_turn>')
            if len(eot_id) == 2 and eot_id[0] == 2:
                self.stop_ids.append(eot_id[1])
        except:
            pass
        if hasattr(self.tokenizer, 'generation_config') and self.tokenizer.generation_config is not None:
            eos_token_id = self.tokenizer.generation_config.eos_token_id
            from collections.abc import Iterable
            if isinstance(eos_token_id, int):
                self.stop_ids.append(eos_token_id)
            elif isinstance(eos_token_id, Iterable):
                for id in eos_token_id:
                    self.stop_ids.append(id)
        self.stop_ids = [stop_id for stop_id in self.stop_ids if stop_id is not None]
        self.stop_ids = list(set(self.stop_ids))
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        if self.tokenizer and hasattr(self.tokenizer, name):
            return getattr(self.tokenizer, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text, **kwargs)

    def _convert_token_to_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def _convert_id_to_token(self, index):
        return self.tokenizer.convert_ids_to_tokens(index)

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def id_to_str(self, token_id):
        try:
            word = self.tokenizer.decode(int(token_id))
        except:
            def contains_replacement(text): return '\uFFFD' in text
            def decode_id(token_id):
                return self.tokenizer.convert_tokens_to_string(
                        self.tokenizer._convert_id_to_token(int(token_id)))
            def decode_ids(token_ids):
                return self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(token_ids))
            word = decode_id(int(token_id))
            # Smollm tokenizer will produce half chinese character, using buffer to decode
            if contains_replacement(word):
                self.decode_buffer.append(token_id)
                buffer_txt = decode_ids(self.decode_buffer)
                if not contains_replacement(buffer_txt):
                    word = buffer_txt
                    self.decode_buffer.clear()
                else:
                    word = ''
        return word

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, model_type, **kwargs):
        return cls(pretrained_model_name_or_path, model_type, **kwargs)

    def apply_chat_template(self, conversation, **kwargs):
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, **kwargs)
        raise RuntimeError('Tokenizer no `apply_chat_template` funtion.')

    def save_vocabulary(self, save_directory, **kwargs):
        file_path = os.path.join(save_directory, "tokenizer.txt")
        # ... (rest of the save_vocabulary logic is unchanged)
        return (file_path,)

    def get_chat_template(self, chat_template = None, tools = None):
        return self.tokenizer.get_chat_template(chat_template, tools)

    def export(self, save_directory, model_path=None, model_type=None):
        """
        Export tokenizer to MNN format with comprehensive tokenizer type support.

        Args:
            save_directory: Directory to save the exported tokenizer
            model_path: Optional model path for tokenizer file discovery
            model_type: Optional model type for special handling

        Returns:
            str: Path to the exported tokenizer file
        """
        import os
        import base64

        # Use provided values or fall back to instance values
        if model_path is None:
            model_path = self.tokenizer_path
        if model_type is None:
            model_type = self.model_type

        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # TOKENIZER MAGIC NUMBER
        MAGIC_NUMBER = 430
        # TOKENIZER TYPE
        SENTENCEPIECE = 0; TIKTOIKEN = 1; BERT = 2; HUGGINGFACE = 3

        def write_line(fp, *args):
            for arg in args:
                for token in arg:
                    fp.write(str(token) + ' ')
            fp.write('\n')

        def write_header(fp, type, speicals, prefix=[]):
            fp.write(f'{MAGIC_NUMBER} {type}\n')
            fp.write(f'{len(speicals)} {len(self.stop_ids)} {len(prefix)}\n')
            write_line(fp, speicals, self.stop_ids, prefix)

        file_path = os.path.join(save_directory, "tokenizer.txt")

        # Collect special tokens from various sources
        special_list = list(self.tokenizer.added_tokens_decoder.keys())
        if hasattr(self.tokenizer, 'special_tokens'):
            for k, v in self.tokenizer.special_tokens.items():
                special_list.append(v)
        if hasattr(self.tokenizer, 'all_special_ids'):
            special_list.extend(self.tokenizer.all_special_ids)
        if hasattr(self.tokenizer, 'gmask_token_id'):
            special_list.append(self.tokenizer.gmask_token_id)

        # Handle generation_config special tokens
        if hasattr(self.tokenizer, 'generation_config') and self.tokenizer.generation_config is not None:
            generation_config = self.tokenizer.generation_config
            if hasattr(generation_config, 'user_token_id'):
                special_list.append(generation_config.user_token_id)
            if hasattr(generation_config, 'assistant_token_id'):
                special_list.append(generation_config.assistant_token_id)

        vocab_list = []
        prefix_list = []

        # Get prefix tokens
        if hasattr(self.tokenizer, 'get_prefix_tokens'):
            prefix_list = self.tokenizer.get_prefix_tokens()

        # Simple prefix token detection
        if len(prefix_list) == 0:
            try:
                test_txt = 'A'
                ids = self.tokenizer.encode(test_txt)
                get_txt = self.tokenizer.decode(ids[-1])
                if len(ids) > 1 and get_txt == test_txt:
                    prefix_list += ids[:-1]
            except Exception:
                pass

        # Load SentencePiece model if available
        sp_model = None
        tokenizer_model = os.path.join(model_path, 'tokenizer.model')
        ice_text_model = os.path.join(model_path, 'ice_text.model')

        try:
            import sentencepiece as spm
            if os.path.exists(tokenizer_model):
                sp_model = spm.SentencePieceProcessor(tokenizer_model)
            elif os.path.exists(ice_text_model):
                sp_model = spm.SentencePieceProcessor(ice_text_model)
        except Exception:
            sp_model = None

        # Check for merge file (BERT/HuggingFace tokenizers)
        merge_file = os.path.join(model_path, 'merges.txt')
        merge_txt = merge_file if os.path.exists(merge_file) else None

        if sp_model is not None:
            # SentencePiece tokenizer export
            NORMAL = 1; UNKNOWN = 2; CONTROL = 3
            USER_DEFINED = 4; UNUSED = 5; BYTE = 6

            for i in range(sp_model.GetPieceSize()):
                token = sp_model.IdToPiece(i)
                score = sp_model.GetScore(i)
                token_type = NORMAL
                if sp_model.IsUnknown(i):
                    token_type = UNKNOWN
                elif sp_model.IsControl(i):
                    token_type = CONTROL
                elif sp_model.IsUnused(i):
                    token_type = UNUSED
                elif sp_model.IsByte(i):
                    token_type = BYTE

                # Handle special cases for specific models
                if model_path == 'Chatglm_6b':
                    if '<n>' in token: token = '\n'
                    if '<|tab|>' in token: token = '\t'
                    if '<|blank_' in token: token = ' ' * int(token[8:token.find('|>')])
                if '▁' in token: token = token.replace('▁', ' ')

                token_encode = base64.b64encode(token.encode("utf-8")).decode("utf8")
                vocab_list.append(f'{token_encode} {score} {token_type}\n')

            # Add special tokens to vocab_list
            for index in special_list:
                if index >= len(vocab_list):
                    try:
                        token = self.tokenizer.decode(index)
                        token_encode = base64.b64encode(token.encode("utf-8")).decode("utf8")
                        vocab_list.append(f'{token_encode} {0} {NORMAL}\n')
                    except:
                        pass

            # Write SentencePiece format
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, SENTENCEPIECE, special_list, prefix_list)
                if model_type == "gemma3" or model_type == "gemma3-text":
                    fp.write(f'{len(vocab_list) + 1}\n')  # +1 for image_soft_token
                else:
                    fp.write(f'{len(vocab_list)}\n')
                for vocab in vocab_list:
                    fp.write(vocab)

        elif hasattr(self.tokenizer, 'mergeable_ranks'):
            # TikToken tokenizer export
            vocab_list = []
            for k, v in self.tokenizer.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + "\n"
                vocab_list.append(line)
            if hasattr(self.tokenizer, 'special_tokens'):
                for k, v in self.tokenizer.special_tokens.items():
                    line = base64.b64encode(k.encode("utf-8")).decode("utf8") + "\n"
                    vocab_list.append(line)
            if hasattr(self.tokenizer, 'added_tokens_decoder'):
                for k, v in self.tokenizer.added_tokens_decoder.items():
                    line = base64.b64encode(v.__str__().encode("utf-8")).decode("utf8") + "\n"
                    vocab_list.append(line)

            # Write TikToken format
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, TIKTOIKEN, special_list, prefix_list)
                fp.write(f'{len(vocab_list)}\n')
                for vocab in vocab_list:
                    fp.write(vocab)

        elif merge_txt is not None:
            # HuggingFace/BERT tokenizer export
            merge_list = []
            vocab = self.tokenizer.get_vocab()
            special_list = list(self.tokenizer.added_tokens_decoder.keys())
            vocab_list = ['<unk>' for i in range(len(vocab))]

            # Load vocab
            for k, v in vocab.items():
                vocab_list[int(v)] = k

            # Load merge
            with open(merge_txt, 'rt') as merge:
                for line in merge.readlines():
                    merge_list.append(line)

            # Write HuggingFace format
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, HUGGINGFACE, special_list)
                fp.write(f'{len(vocab_list)} {len(merge_list)}\n')
                for v in vocab_list:
                    fp.write(v + '\n')
                for m in merge_list:
                    fp.write(m)
        else:
            # Auto-detect tokenizer type and export
            tokenizer_class_name = type(self.tokenizer).__name__.lower()
            vocab = self.tokenizer.get_vocab()

            # Check for SentencePiece-based tokenizers
            if ('xlmroberta' in tokenizer_class_name or
                'roberta' in tokenizer_class_name or
                'sentencepiece' in tokenizer_class_name or
                hasattr(self.tokenizer, 'sp_model') or
                (hasattr(self.tokenizer, 'vocab_file') and
                 self.tokenizer.vocab_file and 'sentencepiece' in self.tokenizer.vocab_file.lower()) or
                # Check for SentencePiece patterns (▁ prefix)
                (len(vocab) > 0 and any('▁' in token for token in list(vocab.keys())[:100]))):
                tokenizer_type = SENTENCEPIECE
                print(f"Detected SentencePiece-based tokenizer: {tokenizer_class_name}")
            elif 'bert' in tokenizer_class_name:
                tokenizer_type = BERT
                print(f"Detected BERT tokenizer: {tokenizer_class_name}")
            else:
                tokenizer_type = TIKTOIKEN
                print(f"Detected TikToken tokenizer: {tokenizer_class_name}")

            vocab = self.tokenizer.get_vocab()

            if tokenizer_type == SENTENCEPIECE:
                # Handle SentencePiece tokenizer
                vocab_list = []
                NORMAL = 1

                for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
                    try:
                        token_bytes = token.encode('utf-8')
                        token_b64 = base64.b64encode(token_bytes).decode('utf-8')
                        vocab_list.append(f'{token_b64} 0.0 {NORMAL}\n')
                    except Exception as e:
                        print(f"Warning: Failed to encode SentencePiece token '{token}': {e}")
                        token_b64 = base64.b64encode('▁'.encode('utf-8')).decode('utf-8')
                        vocab_list.append(f'{token_b64} 0.0 {NORMAL}\n')

                with open(file_path, "w", encoding="utf8") as fp:
                    write_header(fp, SENTENCEPIECE, special_list, prefix_list)
                    fp.write(f'{len(vocab_list)}\n')
                    for vocab_line in vocab_list:
                        fp.write(vocab_line)
            else:
                # Handle BERT or TikToken tokenizer
                def unicode_to_byte(u: int):
                    # Handle special unicode mappings for BERT tokenizers
                    if u >= 256 and u <= 288:
                        return u - 256
                    if u >= 289 and u <= 322:
                        return u - 162
                    if u == 323:
                        return 173
                    return u

                vocab_list = ['<unk>' for i in range(len(vocab))]

                for k, v in vocab.items():
                    if tokenizer_type == BERT:
                        try:
                            vocab_list[int(v)] = k.encode('utf-8')
                        except Exception as e:
                            try:
                                vocab_list[int(v)] = bytes([unicode_to_byte(ord(c)) for c in k])
                            except Exception as e2:
                                print(f"Warning: Failed to encode token '{k}' with id {v}: {e2}")
                                vocab_list[int(v)] = k.encode('utf-8', errors='replace')
                    else:
                        try:
                            vocab_list[int(v)] = bytes([unicode_to_byte(ord(c)) for c in k])
                        except Exception as e2:
                            print(f"Warning: Failed to encode token '{k}' with id {v}: {e2}")
                            vocab_list[int(v)] = k.encode('utf-8', errors='replace')

                with open(file_path, "w", encoding="utf8") as fp:
                    write_header(fp, tokenizer_type, special_list)
                    fp.write(f'{len(vocab_list)}\n')
                    for v in vocab_list:
                        line = base64.b64encode(v).decode("utf8") + "\n"
                        fp.write(line)

        return file_path
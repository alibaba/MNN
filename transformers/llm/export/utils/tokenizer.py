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
        if model_type == 'glm_ocr':
            user_ids = self.tokenizer.encode('<|user|>', add_special_tokens=False)
            if len(user_ids) == 1:
                self.stop_ids.append(user_ids[0])
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
        return (file_path,)

    def get_chat_template(self, chat_template = None, tools = None):
        if chat_template is None and not getattr(self.tokenizer, 'chat_template', None):
            return None
        try:
            return self.tokenizer.get_chat_template(chat_template, tools)
        except ValueError:
            return None

    @staticmethod
    def _generate_nfkc_table():
        import unicodedata
        entries = []
        for cp in range(0x110000):
            try:
                ch = chr(cp)
                normalized = unicodedata.normalize('NFKC', ch)
                if normalized != ch:
                    entries.append((cp, normalized.encode('utf-8')))
            except (ValueError, OverflowError):
                pass
        return entries

    @staticmethod
    def _generate_nfd_table():
        import unicodedata
        entries = []
        for cp in range(0x110000):
            try:
                ch = chr(cp)
                decomposed = unicodedata.normalize('NFD', ch)
                if decomposed != ch:
                    entries.append((cp, decomposed.encode('utf-8')))
            except (ValueError, OverflowError):
                pass
        return entries

    @staticmethod
    def _write_norm_table(fp, entries):
        import struct
        fp.write(struct.pack('<I', len(entries)))
        for cp, utf8 in entries:
            fp.write(struct.pack('<I', cp))
            fp.write(struct.pack('<H', len(utf8)))
            fp.write(utf8)

    def export_mtok(self, save_directory, tokenizer_json_path):
        """Export tokenizer in binary .mtok format (PipelineTokenizer)."""
        import json
        import struct

        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            tj = json.load(f)

        file_path = os.path.join(save_directory, "tokenizer.mtok")
        MAGIC_NUMBER = 430
        PIPELINE = 4

        def pack_str(s):
            if isinstance(s, str):
                s = s.encode('utf-8')
            return struct.pack('<H', len(s)) + s

        with open(file_path, "w", encoding="utf8") as fp:
            # Text header: magic number + type
            fp.write(f'{MAGIC_NUMBER} {PIPELINE}\n')

            # Special tokens info (same as text format)
            special_list = list(set(self.tokenizer.all_special_ids)) if hasattr(self.tokenizer, 'all_special_ids') else []
            if 'added_tokens' in tj:
                for at in tj['added_tokens']:
                    if at.get('special', False) and at.get('id', -1) not in special_list:
                        special_list.append(at['id'])
            special_list = [s for s in special_list if s is not None]

            prefix_list = []
            if hasattr(self.tokenizer, 'get_prefix_tokens'):
                prefix_list = self.tokenizer.get_prefix_tokens()
            if len(prefix_list) == 0:
                try:
                    ids = self.tokenizer.encode('A')
                    get_txt = self.tokenizer.decode(ids[-1])
                    if len(ids) > 1 and get_txt == 'A':
                        prefix_list = ids[:-1]
                except:
                    pass

            fp.write(f'{len(special_list)} {len(self.stop_ids)} {len(prefix_list)}\n')
            tokens_line = ' '.join(str(t) for t in (special_list + self.stop_ids + prefix_list))
            fp.write(tokens_line + '\n' if tokens_line else '\n')

        # Now write binary body
        with open(file_path, "ab") as fp:
            # --- Normalizer ---
            norm = tj.get('normalizer')
            def write_normalizer_bin(fp, norm):
                if norm is None:
                    fp.write(struct.pack('<B', 0))
                    return
                ntype = norm.get('type', '')
                if ntype in ('NFKC', 'Precompiled', 'NFKD'):
                    fp.write(struct.pack('<B', 6))
                    self._write_norm_table(fp, self._generate_nfkc_table())
                elif ntype == 'Prepend':
                    fp.write(struct.pack('<B', 2))
                    fp.write(pack_str(norm.get('prepend', '')))
                elif ntype == 'Replace':
                    fp.write(struct.pack('<B', 3))
                    pattern = ''
                    if isinstance(norm.get('pattern'), dict):
                        pattern = norm['pattern'].get('String', '')
                    elif isinstance(norm.get('pattern'), str):
                        pattern = norm['pattern']
                    fp.write(pack_str(pattern))
                    fp.write(pack_str(norm.get('content', '')))
                elif ntype == 'Sequence':
                    fp.write(struct.pack('<B', 4))
                    normalizers = norm.get('normalizers', [])
                    fp.write(struct.pack('<I', len(normalizers)))
                    for n in normalizers:
                        write_normalizer_bin(fp, n)
                elif ntype == 'BertNormalizer':
                    sa = norm.get('strip_accents', False)
                    # In HuggingFace, strip_accents=None with lowercase=True means strip accents
                    if sa is None and norm.get('lowercase', True):
                        sa = True
                    strip_accents = int(sa or False)
                    if strip_accents:
                        fp.write(struct.pack('<B', 7))
                    else:
                        fp.write(struct.pack('<B', 5))
                    fp.write(struct.pack('<BBBB',
                        int(norm.get('clean_text', True)),
                        int(norm.get('handle_chinese_chars', True)),
                        strip_accents,
                        int(norm.get('lowercase', True))))
                    if strip_accents:
                        self._write_norm_table(fp, self._generate_nfd_table())
                elif ntype == 'Lowercase':
                    fp.write(struct.pack('<B', 5))
                    fp.write(struct.pack('<BBBB', 0, 0, 0, 1))
                elif ntype == 'StripAccents':
                    fp.write(struct.pack('<B', 7))
                    fp.write(struct.pack('<BBBB', 0, 0, 1, 0))
                    self._write_norm_table(fp, self._generate_nfd_table())
                else:
                    fp.write(struct.pack('<B', 0))
            write_normalizer_bin(fp, norm)

            # --- PreTokenizer ---
            pt = tj.get('pre_tokenizer')
            def write_pre_tokenizer_bin(fp, pt):
                if pt is None:
                    fp.write(struct.pack('<B', 0))
                    return
                ptype = pt.get('type', '')
                if ptype == 'ByteLevel':
                    fp.write(struct.pack('<BB', 1, int(pt.get('use_regex', True))))
                elif ptype == 'Digits':
                    fp.write(struct.pack('<BB', 2, int(pt.get('individual_digits', False))))
                elif ptype == 'Metaspace':
                    fp.write(struct.pack('<B', 3))
                    rep = pt.get('replacement', '\u2581')
                    if pt.get('str_rep'):
                        rep = pt['str_rep']
                    fp.write(pack_str(rep))
                    fp.write(struct.pack('<B', int(pt.get('add_prefix_space', True))))
                elif ptype == 'Split':
                    fp.write(struct.pack('<B', 4))
                    pattern = ''
                    if isinstance(pt.get('pattern'), dict):
                        pattern = pt['pattern'].get('Regex', pt['pattern'].get('String', ''))
                    elif isinstance(pt.get('pattern'), str):
                        pattern = pt['pattern']
                    fp.write(pack_str(pattern))
                    behavior = pt.get('behavior', 'Isolated')
                    behavior_id = 0 if behavior == 'Isolated' else (2 if behavior == 'MergedWithPrevious' else 1)
                    fp.write(struct.pack('<BB', int(pt.get('invert', False)), behavior_id))
                elif ptype == 'BertPreTokenizer':
                    fp.write(struct.pack('<B', 5))
                elif ptype == 'Sequence':
                    fp.write(struct.pack('<B', 6))
                    pretokenizers = pt.get('pretokenizers', [])
                    fp.write(struct.pack('<I', len(pretokenizers)))
                    for p in pretokenizers:
                        write_pre_tokenizer_bin(fp, p)
                elif ptype == 'WhitespaceSplit':
                    fp.write(struct.pack('<B', 4))
                    fp.write(pack_str('\\s+'))
                    fp.write(struct.pack('<BB', 0, 1))
                else:
                    fp.write(struct.pack('<B', 0))
            write_pre_tokenizer_bin(fp, pt)

            # --- Model ---
            model = tj.get('model', {})
            mtype = model.get('type', '')
            if not mtype:
                # Infer model type from fields
                if 'continuing_subword_prefix' in model and 'merges' not in model:
                    mtype = 'WordPiece'
                elif isinstance(model.get('vocab'), list):
                    mtype = 'Unigram'
                else:
                    mtype = 'BPE'

            if mtype == 'BPE':
                vocab = model.get('vocab', {})
                merges = model.get('merges', [])
                byte_fallback = int(model.get('byte_fallback', False))

                byte_level = 0
                if pt and pt.get('type') == 'ByteLevel':
                    byte_level = 0
                elif pt and pt.get('type') == 'Sequence':
                    has_bl_pt = any(p.get('type') == 'ByteLevel' for p in pt.get('pretokenizers', []))
                    if not has_bl_pt:
                        dec = tj.get('decoder')
                        if dec and dec.get('type') == 'ByteLevel':
                            byte_level = 1
                        elif dec and dec.get('type') == 'Sequence':
                            if any(d.get('type') == 'ByteLevel' for d in dec.get('decoders', [])):
                                byte_level = 1

                # Sort vocab by token string for binary search in C++
                sorted_vocab = sorted(vocab.items(), key=lambda x: x[0])
                vocab_size = len(sorted_vocab)

                fp.write(struct.pack('<B', 0))  # type=BPE
                fp.write(struct.pack('<I', vocab_size))
                fp.write(struct.pack('<BB', byte_fallback, byte_level))
                fp.write(struct.pack('<I', len(merges)))

                for token, tid in sorted_vocab:
                    fp.write(pack_str(token))
                    fp.write(struct.pack('<I', tid))

                # Build merge pairs with rank, sort by merge_key for binary search in C++
                merge_pairs = []
                for i, m in enumerate(merges):
                    if isinstance(m, str):
                        parts = m.split(' ', 1)
                        if len(parts) == 2:
                            id1 = vocab.get(parts[0], -1)
                            id2 = vocab.get(parts[1], -1)
                            merge_pairs.append((id1, id2, i))
                    elif isinstance(m, list) and len(m) >= 2:
                        id1 = vocab.get(m[0], -1)
                        id2 = vocab.get(m[1], -1)
                        merge_pairs.append((id1, id2, i))
                # Sort by merge_key = (id1 << 32) | id2
                merge_pairs.sort(key=lambda x: (x[0] << 32) | (x[1] & 0xFFFFFFFF))
                for id1, id2, rank in merge_pairs:
                    fp.write(struct.pack('<III', id1, id2, rank))

            elif mtype == 'WordPiece':
                vocab = model.get('vocab', {})
                unk_token = model.get('unk_token', '[UNK]')
                prefix = model.get('continuing_subword_prefix', '##')
                max_chars = model.get('max_input_chars_per_word', 100)
                sorted_vocab = sorted(vocab.items(), key=lambda x: x[0])
                vocab_size = len(sorted_vocab)

                fp.write(struct.pack('<B', 1))  # type=WordPiece
                fp.write(struct.pack('<I', vocab_size))
                fp.write(pack_str(unk_token))
                fp.write(pack_str(prefix))
                fp.write(struct.pack('<I', max_chars))

                for token, tid in sorted_vocab:
                    fp.write(pack_str(token))
                    fp.write(struct.pack('<I', tid))

            elif mtype == 'Unigram':
                vocab = model.get('vocab', [])
                unk_id = model.get('unk_id', 0)
                byte_fallback = int(model.get('byte_fallback', False))

                # Build (token, id, score) and sort by token string
                indexed_vocab = []
                for i, item in enumerate(vocab):
                    if isinstance(item, list) and len(item) >= 2:
                        indexed_vocab.append((item[0], i, item[1]))
                indexed_vocab.sort(key=lambda x: x[0])

                fp.write(struct.pack('<B', 2))  # type=Unigram
                fp.write(struct.pack('<I', len(indexed_vocab)))
                fp.write(struct.pack('<I', unk_id))
                fp.write(struct.pack('<B', byte_fallback))

                for token, tid, score in indexed_vocab:
                    fp.write(pack_str(token))
                    fp.write(struct.pack('<I', tid))
                    fp.write(struct.pack('<d', score))

            # --- Decoder ---
            dec = tj.get('decoder')
            def write_decoder_bin(fp, dec):
                if dec is None:
                    fp.write(struct.pack('<B', 0))
                    return
                dtype = dec.get('type', '')
                if dtype == 'ByteLevel':
                    fp.write(struct.pack('<B', 0))
                elif dtype == 'ByteFallback':
                    fp.write(struct.pack('<B', 1))
                elif dtype == 'Metaspace':
                    fp.write(struct.pack('<B', 2))
                    fp.write(pack_str(dec.get('replacement', '\u2581')))
                    fp.write(struct.pack('<B', int(dec.get('add_prefix_space', True))))
                elif dtype == 'WordPiece':
                    fp.write(struct.pack('<B', 3))
                    fp.write(pack_str(dec.get('prefix', '##')))
                    fp.write(struct.pack('<B', int(dec.get('cleanup', True))))
                elif dtype == 'Fuse':
                    fp.write(struct.pack('<B', 4))
                elif dtype == 'Replace':
                    fp.write(struct.pack('<B', 5))
                    pattern = ''
                    if isinstance(dec.get('pattern'), dict):
                        pattern = dec['pattern'].get('String', '')
                    elif isinstance(dec.get('pattern'), str):
                        pattern = dec['pattern']
                    fp.write(pack_str(pattern))
                    fp.write(pack_str(dec.get('content', '')))
                elif dtype == 'Strip':
                    fp.write(struct.pack('<B', 6))
                    fp.write(pack_str(dec.get('content', '')))
                    fp.write(struct.pack('<II', dec.get('start', 0), dec.get('stop', 0)))
                elif dtype == 'Sequence':
                    fp.write(struct.pack('<B', 7))
                    decoders = dec.get('decoders', [])
                    fp.write(struct.pack('<I', len(decoders)))
                    for d in decoders:
                        write_decoder_bin(fp, d)
                else:
                    fp.write(struct.pack('<B', 0))
            write_decoder_bin(fp, dec)

            # --- Added Tokens ---
            added_tokens = tj.get('added_tokens', [])
            fp.write(struct.pack('<I', len(added_tokens)))
            for at in added_tokens:
                aid = at.get('id', -1)
                special = int(at.get('special', False))
                lstrip = int(at.get('lstrip', False))
                rstrip = int(at.get('rstrip', False))
                content = at.get('content', '')
                fp.write(struct.pack('<I', aid))
                fp.write(struct.pack('<BBB', special, lstrip, rstrip))
                fp.write(pack_str(content))

            # --- Chat Template & Flags ---
            chat_template = ''
            eos_token = ''
            bos_token = ''
            flags = 0
            tokenizer_config_path = os.path.join(os.path.dirname(tokenizer_json_path), 'tokenizer_config.json')
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path, 'r', encoding='utf-8') as tc:
                    tc_json = json.load(tc)
                chat_template = tc_json.get('chat_template', '')
                eos = tc_json.get('eos_token', '')
                if isinstance(eos, dict):
                    eos_token = eos.get('content', '')
                else:
                    eos_token = str(eos) if eos else ''
                bos = tc_json.get('bos_token', '')
                if isinstance(bos, dict):
                    bos_token = bos.get('content', '')
                else:
                    bos_token = str(bos) if bos else ''
                if tc_json.get('clean_up_tokenization_spaces', False) is True:
                    flags |= 0x01
            tpl_bytes = chat_template.encode('utf-8') if chat_template else b''
            eos_bytes = eos_token.encode('utf-8') if eos_token else b''
            fp.write(struct.pack('<I', len(tpl_bytes)))
            fp.write(tpl_bytes)
            fp.write(struct.pack('<H', len(eos_bytes)))
            fp.write(eos_bytes)

            # --- Flags ---
            fp.write(struct.pack('<B', flags))

            # --- BOS token ---
            bos_bytes = bos_token.encode('utf-8') if bos_token else b''
            fp.write(struct.pack('<H', len(bos_bytes)))
            fp.write(bos_bytes)

        return file_path

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

        # Try .mtok format first (pipeline tokenizer) if tokenizer.json exists
        tokenizer_json_path = os.path.join(model_path, 'tokenizer.json')
        if os.path.exists(tokenizer_json_path):
            result = self.export_mtok(save_directory, tokenizer_json_path)
            if result:
                return result

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
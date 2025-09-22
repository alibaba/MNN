import _mnncengine.llm as _F

class Context:
    def __init__(self, llm_obj):
        self._llm_obj = llm_obj
        self._data = self._llm_obj.get_context()

    def refresh(self):
        '''Refresh context data from the underlying C++ object'''
        self._data = self._llm_obj.get_context()

    def update(self, data_dict=None, **kwargs):
        '''Update context data in the underlying C++ object'''
        if data_dict is None:
            data_dict = kwargs
        else:
            data_dict.update(kwargs)

        # Get current context data
        current_data = self._llm_obj.get_context()
        # Update with new data
        current_data.update(data_dict)
        # Set back to C++ object
        self._llm_obj.set_context(current_data)
        # Refresh our local copy
        self._data = current_data

    # Forward parameters
    @property
    def prompt_len(self):
        return self._data.get('prompt_len', 0)

    @prompt_len.setter
    def prompt_len(self, value):
        self.update(prompt_len=value)

    @property
    def gen_seq_len(self):
        return self._data.get('gen_seq_len', 0)

    @gen_seq_len.setter
    def gen_seq_len(self, value):
        self.update(gen_seq_len=value)

    @property
    def all_seq_len(self):
        return self._data.get('all_seq_len', 0)

    @all_seq_len.setter
    def all_seq_len(self, value):
        self.update(all_seq_len=value)

    @property
    def end_with(self):
        return self._data.get('end_with', '')

    @end_with.setter
    def end_with(self, value):
        self.update(end_with=value)

    # Performance metrics (read-only)
    @property
    def load_us(self):
        return self._data.get('load_us', 0)

    @property
    def vision_us(self):
        return self._data.get('vision_us', 0)

    @property
    def audio_us(self):
        return self._data.get('audio_us', 0)

    @property
    def prefill_us(self):
        return self._data.get('prefill_us', 0)

    @property
    def decode_us(self):
        return self._data.get('decode_us', 0)

    @property
    def sample_us(self):
        return self._data.get('sample_us', 0)

    @property
    def pixels_mp(self):
        return self._data.get('pixels_mp', 0.0)

    @property
    def audio_input_s(self):
        return self._data.get('audio_input_s', 0.0)

    # Tokens
    @property
    def current_token(self):
        return self._data.get('current_token', 0)

    @current_token.setter
    def current_token(self, value):
        self.update(current_token=value)

    @property
    def history_tokens(self):
        return self._data.get('history_tokens', [])

    @history_tokens.setter
    def history_tokens(self, value):
        self.update(history_tokens=value)

    @property
    def output_tokens(self):
        return self._data.get('output_tokens', [])

    @output_tokens.setter
    def output_tokens(self, value):
        self.update(output_tokens=value)

    @property
    def generate_str(self):
        return self._data.get('generate_str', '')

    @generate_str.setter
    def generate_str(self, value):
        self.update(generate_str=value)

    def __repr__(self):
        return f"Context({self._data})"

class Llm:

    def __init__(self, c_obj):
        self._c_obj = c_obj
        self._context = Context(self._c_obj)

    def load(self):
        '''
        load model from model_dir

        Parameters
        ----------
        model_dir : model path (split) or model name (single)

        Returns
        -------
        None

        Example:
        -------
        >>> llm.load('../qwen-1.8b-in4/conig.json')
        '''
        self._c_obj.load()

    def apply_chat_template(self, prompt):
        '''
        apply chat template

        Parameters
        ----------
        prompt : input prompt, str or dict

        Returns
        -------
        res : output prompt

        Example:
        -------
        >>> res = llm.apply_chat_template('Hello')
        >>> res = llm.apply_chat_template({'role': 'user', 'content': 'Hello'})
        '''
        return self._c_obj.apply_chat_template(prompt)

    def tokenizer_encode(self, prompt):
        '''
        encode the prompt to token ids

        Parameters
        ----------
        prompt : input prompt, str or dict

        Returns
        -------
        ids : token ids, list of int

        Example:
        -------
        >>> ids = llm.tokenizer_encode('Hello')
        >>> ids = llm.tokenizer_encode({'text': '<img>image_0</img>Describe the image', 'images': [{'data': img_obj, height: 224, width: 224}]})
        '''
        return self._c_obj.tokenizer_encode(prompt)

    def tokenizer_decode(self, token):
        return self._c_obj.tokenizer_decode(token)

    def forward(self, input_ids):
        '''
        forward by input_ids

        Parameters
        ----------
        input_ids : input token ids, list of int

        Returns
        -------
        output_ids : output token ids, list of int

        Example:
        -------
        >>> input_ids = [151644, 872, 198, 108386, 151645, 198, 151644, 77091]
        >>> logits = llm.forward(input_ids)
        '''
        return self._c_obj.forward(input_ids)

    def generate(self, input_ids):
        '''
        generate by input_ids

        Parameters
        ----------
        input_ids : input token ids, list of int

        Returns
        -------
        output_ids : output token ids, list of int

        Example:
        -------
        >>> input_ids = [151644, 872, 198, 108386, 151645, 198, 151644, 77091]
        >>> output_ids = llm.generate(input_ids)
        '''
        return self._c_obj.generate(input_ids)

    def response(self, prompt, stream = False):
        '''
        response by prompt

        Parameters
        ----------
        prompt : input prompt
        stream : generate string stream, default is False

        Returns
        -------
        res : output string

        Example:
        -------
        >>> res = llm.response('Hello', True)
        '''
        return self._c_obj.response(prompt, stream)

    def txt_embedding(self, prompt):
        '''
        get prompt's embedding

        Parameters
        ----------
        prompt : input prompt

        Returns
        -------
        res : embedding var

        Example:
        -------
        >>> res = llm.txt_embedding('Hello')
        '''
        return self._c_obj.txt_embedding(prompt)

    def apply_lora(self, lora_path):
        '''
        apply lora model on base model

        Parameters
        ----------
        lora_path : lora model path

        Returns
        -------
        index : int index

        Example:
        -------
        >>> lora_index = llm.apply_lora('./qwen-1.8b-int4/qwen-1.8b-lora.mnn')
        '''
        return self._c_obj.apply_lora(lora_path)

    def select_module(self, module_index):
        '''
        select current module

        Parameters
        ----------
        module_index : module index

        Returns
        -------
        res : bool

        Example:
        -------
        >>> res = llm.select_module(lora_index)
        '''
        return self._c_obj.select_module(module_index)

    def release_module(self, module_index):
        '''
        release module

        Parameters
        ----------
        module_index : module index, int

        Returns
        -------
        res : bool

        Example:
        -------
        >>> res = llm.release_module(lora_index)
        '''
        return self._c_obj.release_module(module_index)

    def set_config(self, config):
        '''
        set config
        Parameters
        ----------
        config : config dict
        Returns
        -------
        res : bool
        Example:
        -------
        >>> config = {
        >>>     "backend_type": "metal",
        >>> }
        >>> res = llm.set_config(config)
        '''
        import json
        config_str = json.dumps(config)
        return self._c_obj.set_config(config_str)

    def reset(self):
        self._c_obj.reset()

    def stoped(self):
        '''
        Check if the generation has stopped

        Parameters
        ----------
        None

        Returns
        -------
        stopped : bool
            True if generation has stopped, False otherwise

        Example:
        -------
        >>> is_stopped = llm.stoped()
        '''
        return self._c_obj.stoped()

    def generate_init(self, stream=None, end_with=None):
        '''
        Initialize generation with optional stream and end_with parameters

        Parameters
        ----------
        stream : stream object, optional
            Stream for output, defaults to None
        end_with : str, optional
            End token, defaults to None

        Returns
        -------
        None

        Example:
        -------
        >>> llm.generate_init()
        >>> llm.generate_init(None, "\n")
        '''
        self._c_obj.generate_init(stream, end_with)

    @property
    def context(self):
        '''
        Get the Context object associated with this Llm instance

        Parameters
        ----------
        None

        Returns
        -------
        context : Context object

        Example:
        -------
        >>> ctx = llm.context
        >>> print(ctx.prompt_len)
        >>> ctx.prompt_len = 10
        '''
        # Refresh context data from C++ object
        self._context.refresh()
        return self._context

def create(config_path, embedding_model = False):
    '''
    create Llm instance by `config.json`

    Parameters
    ----------
    config_path : config path or model path

    Returns
    -------
    llm : Llm instance

    Example:
    -------
    >>> llm = mllm.create('./qwen-1.8b-int4/config.json')
    '''
    c_obj = _F.create(config_path, embedding_model)
    return Llm(c_obj)
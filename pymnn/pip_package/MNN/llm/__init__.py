import _mnncengine.llm as _F

class Llm:

    def __init__(self, c_obj):
        self._c_obj = c_obj

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

    def tokenizer_encode(self, prompt):
        return self._c_obj.tokenizer_encode(prompt)

    def tokenizer_decode(self, token):
        return self._c_obj.tokenizer_decode(token)

    def forward(self, input_ids):
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
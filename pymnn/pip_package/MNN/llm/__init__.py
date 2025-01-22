import _mnncengine.llm as _F

class LLM(_F.LLM):
    def load(self, model_dir):
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
        super.load(model_dir)

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
        return super.generate(input_ids)

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
        return super.response(prompt, stream)

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
        return super.txt_embedding(prompt)

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
        return super.apply_lora(lora_path)

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
        return super.select_module(module_index)

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
        return super.release_module(module_index)

def create(config_path, embedding_model = False):
    '''
    create LLM instance by `config.json`

    Parameters
    ----------
    config_path : config path or model path

    Returns
    -------
    llm : LLM instance

    Example:
    -------
    >>> llm = mllm.create('./qwen-1.8b-int4/config.json')
    '''
    return _F.create(config_path, embedding_model)
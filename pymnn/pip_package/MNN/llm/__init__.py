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
        >>> output_ids = qwen.generate(input_ids)
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
        >>> res = qwen.response('Hello', True)
        '''
        return super.response(prompt, stream)

def create(config_path):
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
    >>> qwen = llm.create('./qwen-1.8b-int4/config.json')
    '''
    return _F.create(config_path)
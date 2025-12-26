# pip install lm_eval
from lm_eval.models.huggingface import TemplateLM, HFLM
from lm_eval import simple_evaluate

import MNN.llm as mnnllm
import torch
import json
import copy
import argparse
from typing import List

class MNNLM(HFLM):
    def __init__(self, pretrained, batch_size = 1, device = 'cpu'):
        TemplateLM.__init__(self)
        self._model = mnnllm.create(pretrained)
        self._model.load()
        self._model.set_config({'all_logits': True})
        self.backend = "causal"
        self.logits_cache = True
        self.batch_size_per_gpu = batch_size
        self.batch_size_per_gpu = batch_size
        self._max_length = 32768
        self._device = device
        self.pretrained = pretrained
        self.revision = "main"
        self.model_type = "mnnllm"
        self.delta = None
        self.peft = None

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        encoding = self._model.tokenizer_encode(string)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def _model_call(self, inps, attn_mask=None, labels=None):
        # return self._model(inps).logits
        lm_logits_list = []
        for ids in inps:
            ids_list = ids.tolist()
            logits = self.model.forward(ids_list)
            npy_logits = copy.deepcopy(logits.read())
            torch_logits = torch.from_numpy(npy_logits)
            lm_logits_list.append(torch_logits)
        lm_logits = torch.concat(lm_logits_list, axis=0)
        return lm_logits

def eval(model, dataset_task):
    lm = MNNLM(pretrained=model)
    results = simple_evaluate(model=lm, tasks=[dataset_task], batch_size=1, verbosity="ERROR")
    # results = simple_evaluate(model=lm, tasks=["ceval-valid"], batch_size=1)
    # results = simple_evaluate(model=lm, tasks=["hellaswag"], batch_size=1)
    # results = simple_evaluate(model=lm, tasks=["arc_challenge"], batch_size=1)
    # results = simple_evaluate(model=lm, tasks=["arc_challenge","hellaswag","piqa"], batch_size=1)
    filtered_results = results.copy()
    filtered_results = {key: value for key, value in results.items() if key == "results"}
    json_filtered_results = json.dumps(filtered_results, indent=4)
    print(json_filtered_results)
    with open("results.json", "w") as json_file:
        json_file.write(json_filtered_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mnnllm eval', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', type=str, default=None, required=True, help='path to mnn llm model config.')
    parser.add_argument('-d', type=str, default='piqa', required=True, help='tasks to evaluate.')
    args = parser.parse_args()
    eval(args.m, args.d)
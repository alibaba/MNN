import MNN.llm as llm
import sys

if len(sys.argv) < 2:
    print('usage: python llm_example.py <path_to_model_config>')
    exit(1)

config_path = sys.argv[1]
# create model
model = llm.create(config_path)
# load model
model.load()

# response stream
print('>>> Model Status: ', model.context.status)
out = model.response('你好', True)
print('>>> Model Status: ', model.context.status)

# generate
out_ids = model.generate([151644, 872, 198, 108386, 151645, 198, 151644, 77091])
print(out_ids)

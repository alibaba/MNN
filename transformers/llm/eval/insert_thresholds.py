import json


with open('/Users/tgx/Projects/MNN/transformers/llm/eval/thresholds_0.5.json', 'r') as f:
    thresholds_map = json.load(f)

with open('/Users/tgx/Projects/MNN/transformers/llm/export/model/llm.mnn.json', 'r') as f:
    mnn_map = json.load(f)


for key, value in thresholds_map.items():
    for op in mnn_map["oplists"]:
        if op["name"] == key:
            op["main"]["common"]["threshold"] = value


with open('/Users/tgx/Projects/MNN/transformers/llm/export/model/llm.mnn.json', 'w') as f:
    json.dump(mnn_map, f, indent=4)
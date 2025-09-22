import MNN.llm as llm
import MNN.cv as cv
import MNN.numpy as np
import sys


def generate_example(model, prompt):
    prompt['text'] = model.apply_chat_template(prompt['text'])
    ids = model.tokenizer_encode(prompt)
    model.generate_init()
    logits = model.forward(ids)
    token = np.argmax(logits)
    model.context.current_token = token
    word = model.tokenizer_decode(token)
    print(word, end='', flush=True)
    for i in range(128):
        logits = model.forward(token)
        token = np.argmax(logits)
        model.context.current_token = token
        if model.stoped():
            break
        word = model.tokenizer_decode(token)
        print(word, end='', flush=True)

def response_example(model, prompt):
    # response stream
    model.response(prompt, True)
    vision_us = model.context.vision_us
    prefill_us = model.context.prefill_us
    decode_us = model.context.decode_us
    prompt_len = model.context.prompt_len
    decode_len = model.context.gen_seq_len
    pixels_mp = model.context.pixels_mp
    print('pixels : {}'.format(pixels_mp))
    print('vision time : {} ms'.format(vision_us / 1000.0))
    print('prefill speed : {} token/s'.format(prompt_len / (prefill_us / 1000000.0)))
    print('decode speed : {} token/s'.format(decode_len / (decode_us / 1000000.0)))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python vllm_example.py <path_to_model_config>')
        exit(1)

    config_path = sys.argv[1]
    # create model
    model = llm.create(config_path)
    # load model
    model.load()


    img_path = '../../../resource/images/cat.jpg'
    img = cv.imread(img_path)

    prompt = {
        'text': '<img>image_0</img>介绍一下这张图',
        'images': [
            {
                'data': img,
                'height': 420,
                'width': 420
            }
        ]
    }

    # response_example(model, prompt)
    generate_example(model, prompt)
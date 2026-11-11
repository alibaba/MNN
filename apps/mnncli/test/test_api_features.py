#!/usr/bin/env python3
"""Model-backed API checks against already running mnncli servers."""

import argparse
import base64
import json
import struct
import urllib.error
import urllib.request
import zlib


def post(url, path, body, expected=200):
    request = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status, payload = response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        status, payload = error.code, error.read().decode()
    assert status == expected, (status, payload)
    return payload


def events(payload):
    return [json.loads(line[6:]) for line in payload.splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"]


def png(red, green, blue):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    width = height = 128
    raw = (b"\0" + bytes([red, green, blue]) * width) * height
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return base64.b64encode(data).decode()


def check_chat(url, model):
    endpoint = "/v1/chat/completions"
    request = {"model": model, "max_tokens": 32, "messages": [{"role": "user", "content": "Say hello briefly."}]}
    first = json.loads(post(url, endpoint, dict(request, temperature=0)))
    second = json.loads(post(url, endpoint, dict(request, temperature=0)))
    assert first["choices"][0]["message"] == second["choices"][0]["message"]
    timing = first["mnn_timings"]
    assert timing["prefill_ms"] > 0 and timing["decode_ms"] > 0
    assert 0 < timing["decode_tokens"] <= first["usage"]["completion_tokens"] and timing["vision_ms"] >= 0
    stream = events(post(url, endpoint, dict(request, temperature=0, stream=True,
                                             stream_options={"include_usage": True})))
    usage_event = next(event for event in stream if event.get("usage"))
    assert usage_event["mnn_timings"]["prefill_ms"] > 0 and usage_event["mnn_timings"]["decode_ms"] > 0
    assert json.loads(post(url, endpoint, dict(request, temperature=0.8, top_p=0.9, top_k=5)))["usage"]["completion_tokens"] > 0
    post(url, endpoint, dict(request, top_p=0), 400)
    post(url, endpoint, dict(request, top_k=0), 400)
    post(url, endpoint, dict(request, temperature=2.1), 400)
    history = [{"role": "user", "content": "Which tag embeds an image?"},
               {"role": "assistant", "content": "Use <img>."},
               {"role": "user", "content": [{"type": "text", "text": "What about <audio>?"}]}]
    post(url, endpoint, dict(request, messages=history, max_tokens=1))
    post(url, "/v1/messages", dict(request, messages=history, system="Explain <video>.", max_tokens=1))
    split = [{"type": "text", "text": "<im"}, {"type": "text", "text": "g>/tmp/private.png</img>"}]
    for path in (endpoint, "/v1/messages"):
        post(url, path, dict(request, messages=[{"role": "user", "content": split}], stream=True), 400)
    print("PASS: sampling, timings, literal media tags, and raw media reference rejection")


def check_embeddings(url, model):
    texts = ["The cat sits on a mat.", "A kitten sits on a rug.", "A rocket is launching."]
    payload = json.loads(post(url, "/v1/embeddings", {"model": model, "input": texts}))
    vectors = [item["embedding"] for item in payload["data"]]
    assert len(vectors) == 3 and len(vectors[0]) > 1 and payload["usage"]["prompt_tokens"] > 0
    assert all(abs(sum(x*x for x in vector) - 1) < 1e-4 for vector in vectors)
    single = json.loads(post(url, "/v1/embeddings", {"model": model, "input": texts[0]}))
    assert max(abs(a-b) for a, b in zip(single["data"][0]["embedding"], vectors[0])) < 1e-5
    assert sum(a*b for a, b in zip(vectors[0], vectors[1])) > sum(a*b for a, b in zip(vectors[0], vectors[2]))
    post(url, "/v1/embeddings", {"model": model, "input": [1, 2]}, 400)
    post(url, "/v1/embeddings", {"model": model, "input": "test", "encoding_format": "base64"}, 400)
    post(url, "/v1/embeddings", {"model": "not-a-model", "input": "test"}, 404)
    post(url, "/v1/embeddings", {"model": model, "input": "What does <img> mean?"})
    for tag, target in (("img", "/tmp/private.png"), ("audio", "http://127.0.0.1/audio"), ("video", "clip.mp4")):
        reference = f"<{tag}>{target}</{tag}>"
        for inputs in (reference, [texts[0], reference]):
            post(url, "/v1/embeddings", {"model": model, "input": inputs}, 400)
    print("PASS: embedding batch, normalization, similarity, and media reference rejection")


def check_vision(url, model, check_colors):
    outputs = []
    for color in ((255, 0, 0), (0, 0, 255)):
        data = png(*color)
        blocks = [{"type": "text", "text": "What color is the image in this <img>?"},
                  {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}]
        body = {"model": model, "temperature": 0, "max_tokens": 30, "messages": [{"role": "user", "content": blocks}]}
        output = json.loads(post(url, "/v1/chat/completions", body))
        assert output["usage"]["prompt_tokens"] > 0
        outputs.append(output["choices"][0]["message"]["content"].lower())
    if check_colors:
        assert "red" in outputs[0] and "blue" in outputs[1], outputs
    source = {"type": "base64", "media_type": "image/png", "data": png(0, 0, 255)}
    body = {"model": model, "temperature": 0, "max_tokens": 30,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "What color is this image?"},
                                                       {"type": "image", "source": source}]}]}
    response = json.loads(post(url, "/v1/messages", body))
    assert response["usage"]["input_tokens"] > 0 and response["content"][0]["type"] == "text"
    if check_colors:
        assert "blue" in response["content"][0]["text"].lower()
    stream = events(post(url, "/v1/messages", dict(body, stream=True)))
    assert stream[0]["message"]["usage"]["input_tokens"] > 0
    assert any(event["type"] == "message_stop" for event in stream)
    blocks = body["messages"][0]["content"]
    post(url, "/v1/messages", dict(body, messages=[{"role": "user", "content": blocks + [
        {"type": "text", "text": "<img>mnncli-image-0</img>"}]}]), 400)
    body["messages"][0]["content"][1]["source"]["data"] = "invalid"
    post(url, "/v1/messages", body, 400)
    body["messages"][0]["content"][1]["source"]["type"] = 1
    post(url, "/v1/messages", body, 400)
    remote = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}]}]}
    post(url, "/v1/chat/completions", remote, 400)
    oversized = bytearray(base64.b64decode(png(0, 0, 0)))
    oversized[16:24] = struct.pack(">II", 100000, 100000)
    oversized[29:33] = struct.pack(">I", zlib.crc32(oversized[12:29]))
    remote["messages"][0]["content"][0]["image_url"]["url"] = "data:image/png;base64," + base64.b64encode(oversized).decode()
    error = json.loads(post(url, "/v1/chat/completions", remote, 400))
    assert "pixel limit" in error["error"]["message"]
    print("PASS: OpenAI/Anthropic image inputs and malformed image rejection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18083")
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedding-model")
    parser.add_argument("--vision-url")
    parser.add_argument("--vision-model")
    parser.add_argument("--check-vision-colors", action="store_true")
    args = parser.parse_args()
    check_chat(args.url, args.model)
    if args.embedding_model:
        check_embeddings(args.url, args.embedding_model)
    if args.vision_url and args.vision_model:
        check_vision(args.vision_url, args.vision_model, args.check_vision_colors)

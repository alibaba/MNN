// Copyright (c) 2026 Alibaba Group Holding Limited All rights reserved.
#pragma once

namespace mnncli {
constexpr char kChatPage[] = R"HTML(<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MNN Chat</title>
<style>
  :root { color-scheme: light dark; font: 16px/1.5 system-ui, sans-serif; }
  * { box-sizing: border-box; }
  body { margin: auto; max-width: 48rem; height: 100dvh; padding: 0 1rem; display: flex; flex-direction: column; }
  header, .composer { display: flex; align-items: center; gap: .75rem; }
  header { padding: 1rem 0; border-bottom: 1px solid GrayText; flex-wrap: wrap; }
  h1 { font-size: 1.2rem; margin: 0; }
  #model { color: GrayText; overflow-wrap: anywhere; flex: 1; }
  main { flex: 1; overflow: auto; padding: 1rem 0; }
  article { margin-bottom: 1rem; white-space: pre-wrap; overflow-wrap: anywhere; }
  article strong { display: block; font-size: .85rem; }
  form { padding: .75rem 0 1rem; }
  button, textarea, input { font: inherit; padding: .45rem .7rem; border-radius: .4rem; border: 1px solid GrayText; }
  button { cursor: pointer; }
  textarea { flex: 1; min-width: 0; resize: none; }
  #credential { margin-bottom: .6rem; }
  #error { color: CanvasText; margin: .25rem 0; }
</style>
<header><h1>MNN Chat</h1><span id="model"></span><button id="clear" type="button">Clear chat</button></header>
<main id="messages" aria-label="Conversation"></main>
<form id="chat">
  <div id="credential" hidden><label>API key <input name="key" type="password" autocomplete="off"></label></div>
  <div id="error" role="alert"></div>
  <div class="composer"><textarea name="message" rows="2" aria-label="Message" placeholder="Message" required></textarea>
    <button name="send" type="submit">Send</button></div>
</form>
<script>
  const form = document.querySelector('#chat');
  const {key, message, send} = form.elements;
  const messages = document.querySelector('#messages');
  const model = document.querySelector('#model');
  const error = document.querySelector('#error');
  const clear = document.querySelector('#clear');
  const history = [];

  async function request(path, options = {}) {
    const headers = {'Content-Type': 'application/json'};
    if (key.value) headers.Authorization = 'Bearer ' + key.value;
    const response = await fetch(path, {...options, headers});
    if (response.status === 401) document.querySelector('#credential').hidden = false;
    if (!response.ok) throw Error((await response.json()).error.message);
    return response;
  }

  async function loadModel() {
    model.textContent = (await (await request('/api/status')).json()).model;
  }

  function addMessage(label, content) {
    const row = document.createElement('article');
    const name = document.createElement('strong');
    name.textContent = label;
    row.append(name, document.createTextNode(content));
    messages.append(row);
    messages.scrollTop = messages.scrollHeight;
    return row;
  }

  async function readReply(response, row) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = '';
    while (true) {
      const {value, done} = await reader.read();
      if (done) throw Error('The reply was interrupted.');
      pending += decoder.decode(value, {stream: true});
      const lines = pending.split('\n');
      pending = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (data === '[DONE]') return row.lastChild.data;
        const chunk = JSON.parse(data);
        if (chunk.error) throw Error(chunk.error.message);
        const content = chunk.choices[0]?.delta.content;
        if (content) {
          row.lastChild.appendData(content);
          messages.scrollTop = messages.scrollHeight;
        }
      }
    }
  }

  clear.onclick = () => {
    history.length = 0;
    messages.replaceChildren();
    error.textContent = '';
    message.focus();
  };
  message.onkeydown = event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (!send.disabled) form.requestSubmit();
    }
  };
  form.onsubmit = async event => {
    event.preventDefault();
    const content = message.value.trim();
    if (!content || send.disabled) return;
    send.disabled = clear.disabled = message.disabled = true;
    error.textContent = message.value = '';
    const user = addMessage('You', content);
    const reply = addMessage('Assistant', '');
    try {
      if (!model.textContent) await loadModel();
      const turn = {role: 'user', content};
      const response = await request('/v1/chat/completions', {method: 'POST',
        body: JSON.stringify({model: model.textContent, messages: [...history, turn], stream: true})});
      history.push(turn, {role: 'assistant', content: await readReply(response, reply)});
    } catch (err) {
      user.remove();
      reply.remove();
      message.value = content;
      error.textContent = err.message;
    } finally {
      send.disabled = clear.disabled = message.disabled = false;
      message.focus();
    }
  };
  loadModel().catch(err => {
    if (document.querySelector('#credential').hidden) error.textContent = err.message;
  });
</script>
</html>)HTML";
} // namespace mnncli

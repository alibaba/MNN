// Exercises the Studio tool loop against a running model and agent server.
// Only a harmless printf command in an isolated directory is automatically approved.
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const vm = require('vm');
const {TextDecoder} = require('util');

const url = process.argv[2] || 'http://127.0.0.1:18089';
const model = process.argv[3] || 'Qwen3.5-4B-MNN';
const html = fs.readFileSync(path.join(__dirname, '../web/index.html'), 'utf8');
function section(start, end) {
  const first = html.indexOf(start);
  const last = html.indexOf(end, first);
  assert(first >= 0 && last > first);
  return html.slice(first, last);
}

async function post(endpoint, body) {
  const response = await fetch(url + '/api/agent/' + endpoint, {
    method:'POST', headers:{'Content-Type':'application/json', Origin:url}, body:JSON.stringify(body)
  });
  const result = await response.json();
  assert(response.ok, JSON.stringify(result));
  return result;
}

async function main() {
  const workspace = fs.mkdtempSync(path.join(os.tmpdir(), 'mnncli-web-live-'));
  let session;
  try {
    session = (await post('session', {workspace, permissions:{command:true}})).session_id;
    const chat = {id:'test', model, messages:[]};
    const state = {status:{model, tools:true}, toolsEnabled:true, agentSession:session,
      agentPermissions:{command:true}, mcpTools:{}, settings:{maxTokens:'256',temperature:'',topP:''},
      chats:[chat], active:'test', images:[], busy:false, controller:null};
    const elements = Object.fromEntries(['prompt','composerStatus','approvalDialog','approvalPreview',
      'approvalWarning','approvalAllow','approvalDeny'].map(name => [name, {}]));
    elements.prompt.value = '请使用 run_command 工具执行 printf mnn_tool_ok 并报告命令实际输出。';
    const approvals = [];
    elements.approvalDialog.showModal = function() {
      this.open = true;
      const action = JSON.parse(elements.approvalPreview.textContent);
      const command = action.arguments.command;
      const allow = action.tool === 'run_command' && action.workspace === fs.realpathSync(workspace) &&
        /^printf (?:mnn_tool_ok|'mnn_tool_ok'|"mnn_tool_ok")$/.test(command);
      approvals.push({command, allow});
      setImmediate(() => elements[allow ? 'approvalAllow' : 'approvalDeny'].onclick());
    };
    elements.approvalDialog.close = function() { this.open = false; };
    const browserFetch = (route, options) => fetch(url + route, {
      ...options, headers:{...options.headers, Origin:url}
    });
    const context = {state, $:id => elements[id], current:() => chat, id:() => chat.id,
      toast:() => {}, renderAttachments:() => {}, renderHistory:() => {}, renderChat:() => {},
      renderStatus:() => {}, saveChat:async() => {}, authHeaders:() => ({}), fetch:browserFetch,
      AbortController, TextDecoder, Intl, Date, JSON, Number, String, Error, Promise};
    vm.createContext(context);
    const script = section('  const builtinTools=', '  const appearanceKey') +
      section('  async function agentPost', '  function toast') +
      section('  function apiMessages(chat)', '  function showSettings()');
    const send = vm.runInContext(script + '\nsend', context);
    await send();
    assert(approvals.some(approval => approval.allow), JSON.stringify({approvals, messages:chat.messages}));
    const result = chat.messages.find(message => message.role === 'tool');
    assert(result && JSON.parse(result.content).output === 'mnn_tool_ok', JSON.stringify(chat.messages));
    assert(chat.messages.at(-1).content.includes('mnn_tool_ok'), JSON.stringify(chat.messages));
    console.log('PASS: Studio streaming, approval, real command and model response');
  } finally {
    if (session) await post('close', {session_id:session});
    fs.rmSync(workspace, {recursive:true, force:true});
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });

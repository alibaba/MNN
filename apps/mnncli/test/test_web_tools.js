const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const {TextDecoder, TextEncoder} = require('util');

const html=fs.readFileSync(path.join(__dirname,'../web/index.html'),'utf8');
function section(start,end) {const a=html.indexOf(start),b=html.indexOf(end,a);assert(a>=0 && b>a);return html.slice(a,b);}
const script=section('  const builtinTools=','  const appearanceKey')+section('  function apiMessages(chat)','  function showSettings()');
const stream=events=>{const bytes=new TextEncoder().encode(events.map(event=>'data: '+(typeof event==='string'?event:JSON.stringify(event))+'\n\n').join(''));let offset=0;return {getReader(){return {async read(){if(offset>=bytes.length)return {done:true};const value=bytes.slice(offset,offset+37);offset+=37;return {done:false,value};}};}};};

async function runScenario({denied=false, mcp=false}={}) {
  const chat={id:'test',model:'model',messages:[],updated:0};
  const state={status:{model:'model',tools:true},toolsEnabled:true,agentSession:'session',agentPermissions:{command:true,read:true,write:false},
    mcpTools:mcp?{fixture:[{type:'function',function:{name:'mcp_fixture_echo_123',description:'echo',parameters:{type:'object'}}}]}:{},
    settings:{maxTokens:'',temperature:'',topP:''},chats:[chat],active:'test',images:[],busy:false,controller:null};
  const elements=Object.fromEntries(['prompt','composerStatus','attachButton'].map(id=>[id,{value:'',textContent:''}]));
  elements.prompt.value='run a tool';
  const requests=[],invocations=[];
  const toolName=mcp?'mcp_fixture_echo_123':'run_command', toolArgs=mcp?{text:'hello'}:{command:'pwd'};
  const context={state,$:id=>elements[id],current:()=>chat,id:()=>chat.id,toast:()=>{},renderAttachments:()=>{},renderHistory:()=>{},
    renderChat:()=>{},renderStatus:()=>{},saveChat:async()=>{},authHeaders:()=>({}),AbortController,TextDecoder,Intl,Date,JSON,Number,String,Error,
    invokeAgentTool:async(name,args)=>{invocations.push({name,args});if(denied){state.agentDenied=true;return JSON.stringify({error:'denied'});}return JSON.stringify({output:'ok'});},
    fetch:async(_,options)=>{const body=JSON.parse(options.body);requests.push(body);
      if(requests.length===1)return {ok:true,body:stream([{choices:[{delta:{tool_calls:[{index:0,id:'call_1',function:{name:toolName,arguments:JSON.stringify(toolArgs)}}]}}]},'[DONE]'])};
      return {ok:true,body:stream([{choices:[{delta:{content:denied?'Denied.':'Done.'}}]},
        {choices:[],usage:{prompt_tokens:10,completion_tokens:4},mnn_timings:{prefill_ms:50,decode_ms:100,decode_tokens:3,vision_ms:0}},'[DONE]'])};}};
  vm.createContext(context);
  const api=vm.runInContext(script+'\n({send,calculate,executeTool})',context);
  assert.strictEqual(api.calculate('-(2+3)*4/2'),-10);
  assert.throws(()=>api.calculate('1/0'));
  assert.throws(()=>api.calculate('1+globalThis.process.exit()'));
  await api.send();
  assert.strictEqual(invocations.length,1);
  assert.strictEqual(invocations[0].name,toolName);
  assert.deepStrictEqual(JSON.parse(JSON.stringify(invocations[0].args)),toolArgs);
  const names=requests[0].tools.map(tool=>tool.function.name);
  assert(names.includes('run_command') && names.includes('read_file') && names.includes('list_directory'));
  assert(!names.includes('write_file') && !names.some(name=>name.startsWith('github_')));
  assert.strictEqual(names.includes('mcp_fixture_echo_123'),mcp);
  assert.strictEqual(requests[1].messages.find(message=>message.role==='tool').tool_call_id,'call_1');
  if(denied) assert(!requests[1].tools);
  assert.strictEqual(chat.messages.at(-1).content,denied?'Denied.':'Done.');
  assert.strictEqual(state.busy,false);
}

async function approvalScenario(allowed) {
  const calls=[];
  const state={agentSession:'session',approvalPending:false,agentDenied:false};
  const dialog={open:false,showModal(){this.open=true;},close(){this.open=false;}};
  const elements={approvalDialog:dialog,approvalPreview:{},approvalWarning:{},approvalAllow:{},approvalDeny:{}};
  const fetch=async(url,options)=>{calls.push({url,body:JSON.parse(options.body)});
    const data=url.endsWith('/prepare')?{request_id:'id',action:{tool:'run_command',workspace:'/tmp/test',arguments:{command:'pwd'}}}:{output:'ok'};
    return {ok:true,json:async()=>data};};
  const context={state,$:id=>elements[id],authHeaders:()=>({}),fetch,JSON,Promise,Error};
  vm.createContext(context);
  const api=vm.runInContext(section('  async function agentPost','  function toast')+'\n({invokeAgentTool})',context);
  const pending=api.invokeAgentTool('run_command',{command:'pwd'});
  await new Promise(resolve=>setImmediate(resolve));
  assert.strictEqual(dialog.open,true);
  assert.strictEqual(calls.length,1);
  assert(elements.approvalPreview.textContent.includes('pwd'));
  (allowed?elements.approvalAllow:elements.approvalDeny).onclick();
  const result=JSON.parse(await pending);
  assert.strictEqual(calls[1].url.endsWith(allowed?'/execute':'/cancel'),true);
  assert.strictEqual(!!result.error,!allowed);
  assert.strictEqual(state.agentDenied,!allowed);
}

async function settingsScenario() {
  const labels=vm.runInNewContext(section('  function toolAvailability(state)', '  function renderStatus()')+'\ntoolAvailability');
  const state={agentSession:null,agentPermissions:{},mcpTools:{},mcpServers:[],toolsEnabled:false};
  assert.strictEqual(labels(state),'仅时间/计算器 · 设置权限');
  const elements=Object.fromEntries(['agentForm','agentRead','agentWrite','agentCommand','agentMcp','agentWorkspace','agentMcpConfig'].map(id=>[id,{}]));
  elements.agentRead.checked=true;elements.agentCommand.checked=true;
  elements.agentWorkspace.value='/tmp/workspace';elements.agentMcpConfig.value='';
  const saved={},requests=[];
  const context={state,$:id=>elements[id],toolsKey:'tools',localStorage:{setItem:(key,value)=>{saved[key]=value;}},
    agentPost:async(path,body)=>{requests.push({path,body});return {session_id:'new-session',permissions:body.permissions,mcp_servers:[]};},
    renderAgentSettings:()=>{},renderStatus:()=>{},toast:()=>{},JSON};
  vm.runInNewContext(section("  $('agentForm').onsubmit=", "  $('agentDisable').onclick="),context);
  await elements.agentForm.onsubmit({preventDefault(){}});
  assert.strictEqual(state.toolsEnabled,true);
  assert.strictEqual(saved.tools,'true');
  assert.strictEqual(state.agentSession,'new-session');
  assert.strictEqual(labels(state),'命令、文件已启用');
  assert.strictEqual(requests[0].body.permissions.command,true);
  state.agentPermissions.command=false;
  assert.strictEqual(labels(state),'文件已启用 · 命令未启用');
  state.agentSession=null;
  assert.strictEqual(labels(state),'仅时间/计算器 · 设置权限');
}

async function main(){await runScenario();await runScenario({denied:true});await runScenario({mcp:true});await approvalScenario(true);await approvalScenario(false);await settingsScenario();console.log('PASS: tool exposure, session settings, streamed calls, approval, denial and MCP dispatch');}
main().catch(error=>{console.error(error);process.exitCode=1;});

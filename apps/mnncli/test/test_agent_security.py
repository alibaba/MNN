#!/usr/bin/env python3
"""Model-backed tests for local agent authorization and workspace boundaries."""

import argparse
import json
import os
import pathlib
import sys
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--port', type=int, default=18093)
    args = parser.parse_args()
    binary = pathlib.Path(__file__).resolve().parents[1] / 'build_mnncli/mnncli'
    origin = f'http://127.0.0.1:{args.port}'
    with tempfile.TemporaryDirectory(prefix='mnncli-agent-test-') as temp:
        temp = pathlib.Path(temp)
        workspace = temp / 'workspace'
        workspace.mkdir()
        (workspace / 'hello.txt').write_text('hello world')
        (workspace / 'escape').symlink_to(temp, target_is_directory=True)
        (temp / 'outside.txt').write_text('outside')
        with open(temp / 'server.log', 'w') as log:
            process = subprocess.Popen([str(binary), 'serve', 'test', '--config', args.config, '--port', str(args.port),
                                        '--api-key', 'agent-test'], cwd=temp, stdout=log, stderr=log)
            try:
                for _ in range(120):
                    try:
                        with urllib.request.urlopen(urllib.request.Request(origin + '/api/status', headers={'Authorization':'Bearer agent-test'}), timeout=1) as response:
                            assert json.load(response)['agent_tools']
                        break
                    except (OSError, urllib.error.URLError):
                        if process.poll() is not None:
                            raise RuntimeError('Server exited while starting')
                        time.sleep(0.25)
                else:
                    raise RuntimeError('Server did not start')

                def post(path, body, headers=None):
                    headers = {'Content-Type':'application/json', 'Origin':origin, 'Authorization':'Bearer agent-test', **(headers or {})}
                    request = urllib.request.Request(origin + path, json.dumps(body).encode(), headers)
                    try:
                        with urllib.request.urlopen(request, timeout=20) as response:
                            return response.status, json.load(response)
                    except urllib.error.HTTPError as error:
                        return error.code, json.load(error)

                endpoint = '/api/agent/'
                options = {'workspace':str(workspace), 'permissions':{'read':True, 'write':True, 'command':True}}
                assert post(endpoint+'session', options, {'Origin':'https://evil.example'})[0] == 403
                assert post(endpoint+'session', options, {'Authorization':'Bearer wrong'})[0] == 401
                status, data = post(endpoint+'session', options)
                assert status == 200
                session = data['session_id']
                status, other = post(endpoint+'session', options)
                assert status == 200

                def prepare(tool, arguments, session_id=session):
                    return post(endpoint+'prepare', {'session_id':session_id, 'tool':tool, 'arguments':arguments})

                def execute(request_id, session_id=session, **extra):
                    return post(endpoint+'execute', {'session_id':session_id, 'request_id':request_id, **extra})

                assert prepare('read_file', {'path':'../outside.txt'})[0] == 400
                assert prepare('run_command', {'command':'echo x', 'cwd':'../'})[0] == 400
                status, req = prepare('read_file', {'path':'escape/outside.txt'})
                assert status == 200 and execute(req['request_id'])[0] == 400
                status, req = prepare('read_file', {'path':'hello.txt'})
                assert status == 200 and execute(req['request_id'], other['session_id'])[0] == 400
                status, data = execute(req['request_id'], arguments={'path':'../outside.txt'})
                assert status == 200 and data['content'] == 'hello world'
                assert execute(req['request_id'])[0] == 400
                status, req = prepare('write_file', {'path':'new.txt', 'content':'approved'})
                assert status == 200 and not (workspace / 'new.txt').exists()
                assert post(endpoint+'cancel', {'session_id':session, 'request_id':req['request_id']})[0] == 200
                assert execute(req['request_id'])[0] == 400 and not (workspace / 'new.txt').exists()
                status, req = prepare('write_file', {'path':'new.txt', 'content':'approved'})
                assert status == 200 and execute(req['request_id'])[0] == 200
                assert (workspace / 'new.txt').read_text() == 'approved'
                status, req = prepare('write_file', {'path':'hello.txt', 'content':'changed'})
                assert status == 200 and execute(req['request_id'])[0] == 400
                assert (workspace / 'hello.txt').read_text() == 'hello world'
                status, req = prepare('write_file', {'path':'hello.txt', 'content':'changed', 'overwrite':True})
                assert status == 200 and execute(req['request_id'])[0] == 200
                assert (workspace / 'hello.txt').read_text() == 'changed'
                status, req = prepare('run_command', {'command':'printf output; printf error >&2; exit 7'})
                assert status == 200
                status, result = execute(req['request_id'])
                assert status == 200 and result['exit_code'] == 7 and 'output' in result['output'] and 'error' in result['output']
                status, req = prepare('run_command', {'command':'sleep 30 & wait', 'timeout_seconds':1})
                assert status == 200 and execute(req['request_id'])[1]['timed_out']
                status, restricted = post(endpoint+'session', {'workspace':str(workspace), 'permissions':{'read':True}})
                assert status == 200 and prepare('run_command', {'command':'pwd'}, restricted['session_id'])[0] == 400
                marker = temp / 'mcp-started'
                fixture = pathlib.Path(__file__).resolve().parent / 'fixtures/mcp_test_server.py'
                config = temp / 'mcp.json'
                config.write_text(json.dumps({'mcpServers':{'fixture':{'command':sys.executable, 'args':[str(fixture)],
                    'env':{'MNN_MCP_TEST_MARKER':str(marker),'MNN_MCP_TEST_SECRET':'private-secret'}}}}))
                status, mcp = post(endpoint+'session', {'workspace':str(workspace),'permissions':{'mcp':True},'mcp_config':str(config)})
                assert status == 200 and mcp['mcp_servers'][0]['name'] == 'fixture'
                assert 'private-secret' not in json.dumps(mcp)
                mcp_session = mcp['session_id']
                status, request = prepare('mcp_connect', {'server':'fixture'}, mcp_session)
                assert status == 200 and not marker.exists()
                assert 'private-secret' not in json.dumps(request) and request['action']['arguments']['config']['env']['MNN_MCP_TEST_SECRET'] == '[hidden]'
                status, connected = execute(request['request_id'], mcp_session)
                assert status == 200 and marker.exists() and len(connected['tools']) == 2
                tool = next(item['function']['name'] for item in connected['tools'] if '/echo]' in item['function']['description'])
                assert prepare('mcp_not_registered', {}, mcp_session)[0] == 400
                status, request = prepare(tool, {'text':'hello'}, mcp_session)
                assert status == 200 and request['action']['arguments']['name'] == 'echo'
                post(endpoint+'cancel', {'session_id':mcp_session, 'request_id':request['request_id']})
                assert execute(request['request_id'], mcp_session)[0] == 400
                for count in (1, 2):
                    status, request = prepare(tool, {'text':'hello'}, mcp_session)
                    assert status == 200
                    status, result = execute(request['request_id'], mcp_session)
                    assert status == 200 and result['content'][0]['text'] == 'hello'
                    assert result['structuredContent'] == {'calls':count,'secret_present':True}
                assert post(endpoint+'mcp/disconnect', {'session_id':mcp_session,'server':'fixture'})[0] == 200
                assert prepare(tool, {'text':'hello'}, mcp_session)[0] == 400
                assert post(endpoint+'close', {'session_id':mcp_session})[0] == 200
                http_port, http_started, http_closed = (temp / name for name in ('http-port', 'http-started', 'http-closed'))
                http_fixture = pathlib.Path(__file__).resolve().parent / 'fixtures/mcp_http_test_server.py'
                env = {**os.environ, 'MNN_MCP_HTTP_PORT':str(http_port),
                       'MNN_MCP_HTTP_STARTED':str(http_started), 'MNN_MCP_HTTP_CLOSED':str(http_closed)}
                http_process = subprocess.Popen([sys.executable, str(http_fixture)], env=env)
                try:
                    for _ in range(100):
                        if http_port.exists() and http_port.read_text():
                            break
                        time.sleep(0.05)
                    assert http_port.exists()
                    config.write_text(json.dumps({'mcpServers':{'remote':{
                        'url':f'http://127.0.0.1:{http_port.read_text()}/mcp',
                        'headers':{'X-Test-Secret':'http-private-secret'}}}}))
                    status, remote = post(endpoint+'session', {'workspace':str(workspace),
                        'permissions':{'mcp':True}, 'mcp_config':str(config)})
                    assert status == 200 and 'http-private-secret' not in json.dumps(remote)
                    remote_session = remote['session_id']
                    status, request = prepare('mcp_connect', {'server':'remote'}, remote_session)
                    assert status == 200 and not http_started.exists()
                    assert 'http-private-secret' not in json.dumps(request)
                    assert request['action']['arguments']['config']['headers']['X-Test-Secret'] == '[hidden]'
                    status, connected = execute(request['request_id'], remote_session)
                    assert status == 200, connected
                    assert http_started.exists() and len(connected['tools']) == 1
                    tool = connected['tools'][0]['function']['name']
                    for count in (1, 2):
                        status, request = prepare(tool, {'text':'hello HTTP'}, remote_session)
                        assert status == 200
                        status, result = execute(request['request_id'], remote_session)
                        assert status == 200 and result['content'][0]['text'] == 'hello HTTP'
                        assert result['structuredContent']['calls'] == count
                    assert post(endpoint+'mcp/disconnect', {'session_id':remote_session,'server':'remote'})[0] == 200
                    assert http_closed.exists()
                    assert post(endpoint+'close', {'session_id':remote_session})[0] == 200

                    delete_started = temp / 'http-delete-started'
                    # A slow MCP shutdown must not block unrelated agent sessions.
                    http_process.terminate()
                    http_process.wait(timeout=5)
                    http_port.unlink()
                    env['MNN_MCP_HTTP_DELETE_STARTED'] = str(delete_started)
                    http_process = subprocess.Popen([sys.executable, str(http_fixture)], env=env)
                    for _ in range(100):
                        if http_port.exists() and http_port.read_text():
                            break
                        time.sleep(0.05)
                    config.write_text(json.dumps({'mcpServers':{'remote':{
                        'url':f'http://127.0.0.1:{http_port.read_text()}/mcp',
                        'headers':{'X-Test-Secret':'http-private-secret'}}}}))
                    status, remote = post(endpoint+'session', {'workspace':str(workspace),
                        'permissions':{'mcp':True}, 'mcp_config':str(config)})
                    remote_session = remote['session_id']
                    status, request = prepare('mcp_connect', {'server':'remote'}, remote_session)
                    assert status == 200 and execute(request['request_id'], remote_session)[0] == 200
                    close_result = []
                    closer = threading.Thread(target=lambda: close_result.append(
                        post(endpoint+'close', {'session_id':remote_session})[0]))
                    closer.start()
                    for _ in range(100):
                        if delete_started.exists():
                            break
                        time.sleep(0.02)
                    assert delete_started.exists()
                    assert prepare('read_file', {'path':'hello.txt'})[0] == 200
                    assert closer.is_alive(), 'Closing MCP blocked another session'
                    closer.join(timeout=10)
                    assert close_result == [200]
                    for server in ({'url':'http://example.com/mcp'},
                                   {'url':'http://127.0.0.1/mcp', 'headers':{'Host':'other'}}):
                        config.write_text(json.dumps({'mcpServers':{'bad':server}}))
                        assert post(endpoint+'session', {'workspace':str(workspace),
                            'permissions':{'mcp':True}, 'mcp_config':str(config)})[0] == 400
                finally:
                    http_process.terminate()
                    http_process.wait(timeout=5)
                assert post(endpoint+'close', {'session_id':session})[0] == 200
                assert prepare('read_file', {'path':'hello.txt'})[0] == 400
                print('PASS: agent authorization, workspace, commands, stdio and Streamable HTTP MCP')
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    main()
